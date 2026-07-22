#!/usr/bin/env python3
"""Test whether a frozen policy-Q target can improve the deployed CartPole actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import learned_continue_discount, symexp_twohot_bins
from dreamer.models.dreaming import estimate_policy_lambda_action_values

if __package__:
    from scripts.probe_cartpole_actor_supervision import evaluate_actor, observe_latent
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_actor_supervision import evaluate_actor, observe_latent  # type: ignore[import-not-found]
    from probe_cartpole_q import load_checkpoint_models  # type: ignore[import-not-found]


def confident_policy_label(
    means: torch.Tensor,
    standard_errors: torch.Tensor,
    *,
    z_score: float = 1.96,
) -> tuple[int, float, float] | None:
    """Return a binary preference when the Monte Carlo difference is separated."""
    if means.numel() != 2 or standard_errors.numel() != 2:
        raise ValueError("CartPole policy labels require exactly two actions")
    delta = float((means[1] - means[0]).item())
    delta_se = float(torch.sqrt(standard_errors[0] ** 2 + standard_errors[1] ** 2).item())
    if abs(delta) <= float(z_score) * delta_se:
        return None
    return int(delta > 0.0), delta, delta_se


def deterministic_split(
    count: int, validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic, disjoint train and validation indices."""
    if count < 2:
        raise ValueError("at least two labels are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(count, generator=generator)
    validation_count = max(1, min(count - 1, round(count * validation_fraction)))
    return order[validation_count:], order[:validation_count]


@torch.no_grad()
def collect_policy_dataset(
    cfg,
    actor,
    critic,
    encoder,
    world_model,
    *,
    device: str,
    episodes: int,
    seed: int,
    policy_q_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float], int]:
    """Follow the deployed actor and retain statistically separated policy-Q labels."""
    env = gym.make(cfg.environment_name)
    bins = symexp_twohot_bins(
        cfg.b_start,
        cfg.b_end,
        int(cfg.num_bins),
        device=device,
        dtype=torch.float32,
    )
    imagination_discount = learned_continue_discount(
        cfg.gamma, bool(getattr(cfg, "contdisc", True))
    )
    generator = torch.Generator(device=device).manual_seed(seed + 3_000_000)
    latents: list[torch.Tensor] = []
    labels: list[int] = []
    deltas: list[float] = []
    delta_standard_errors: list[float] = []
    visited_states = 0
    h_prev_backup = world_model.h_prev.clone()

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            previous_action = torch.zeros(1, cfg.n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))

            while True:
                state = np.asarray(obs, dtype=np.float32)
                h_z, h, z_embed = observe_latent(
                    encoder,
                    world_model,
                    cfg,
                    state,
                    h,
                    previous_action,
                    z_prev_embed,
                )
                state_h_prev = world_model.h_prev.clone()
                try:
                    means, standard_errors = estimate_policy_lambda_action_values(
                        h_z,
                        z_embed,
                        actor,
                        critic,
                        world_model,
                        cfg.n_actions,
                        cfg.d_hidden,
                        bins,
                        imagination_discount,
                        cfg.lam,
                        cfg.num_dream_steps,
                        policy_q_samples,
                        generator=generator,
                        terminal_reward_penalty=float(
                            getattr(cfg, "terminal_reward_penalty", 0.0)
                        ),
                    )
                finally:
                    world_model.h_prev = state_h_prev

                label = confident_policy_label(means[0], standard_errors[0])
                if label is not None:
                    action_label, delta, delta_se = label
                    latents.append(h_z.squeeze(0).detach().cpu())
                    labels.append(action_label)
                    deltas.append(delta)
                    delta_standard_errors.append(delta_se)

                action = int(actor(h_z).argmax(dim=-1).item())
                obs, _reward, terminated, truncated, _ = env.step(action)
                previous_action = F.one_hot(
                    torch.tensor([action], device=device), num_classes=cfg.n_actions
                ).float()
                z_prev_embed = z_embed.detach()
                visited_states += 1
                if terminated or truncated:
                    break
    finally:
        env.close()
        world_model.h_prev = h_prev_backup

    if len(latents) < 2:
        raise RuntimeError("policy-Q collection produced fewer than two confident labels")
    return (
        torch.stack(latents),
        torch.tensor(labels, dtype=torch.long),
        deltas,
        delta_standard_errors,
        visited_states,
    )


def fit_actor(
    actor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    """Fine-tune an actor copy against frozen policy-Q labels."""
    torch.manual_seed(seed)
    if train_x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr)
    losses: list[float] = []
    actor.train()
    for _epoch in range(epochs):
        order = torch.randperm(train_x.shape[0], device=train_x.device)
        for start in range(0, train_x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            loss = F.cross_entropy(actor(train_x[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    actor.eval()
    with torch.no_grad():
        train_accuracy = float(
            (actor(train_x).argmax(dim=-1) == train_y).float().mean().cpu()
        )
        validation_accuracy = float(
            (actor(validation_x).argmax(dim=-1) == validation_y)
            .float()
            .mean()
            .cpu()
        )
    return {
        "final_loss": losses[-1],
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy-q-samples", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error("episode counts must be positive")
    if args.policy_q_samples < 2:
        parser.error("--policy-q-samples must be at least 2")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("actor fit settings must be positive")

    device = resolve_device(args.device)
    cfg, actor, critic, _q_critic, encoder, world_model, checkpoint, critic_key, _q_key = (
        load_checkpoint_models(args.checkpoint.resolve(), device)
    )
    del checkpoint
    original_eval = evaluate_actor(
        cfg,
        actor,
        encoder,
        world_model,
        device,
        episodes=args.eval_episodes,
        seed=args.seed,
    )
    xs, ys, deltas, delta_ses, visited_states = collect_policy_dataset(
        cfg,
        actor,
        critic,
        encoder,
        world_model,
        device=device,
        episodes=args.episodes,
        seed=args.seed,
        policy_q_samples=args.policy_q_samples,
    )
    train_indices, validation_indices = deterministic_split(
        len(ys), args.validation_fraction, args.seed + 4_000_000
    )
    xs = xs.to(device)
    ys = ys.to(device)
    fit_summary = fit_actor(
        actor,
        xs[train_indices],
        ys[train_indices],
        xs[validation_indices],
        ys[validation_indices],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 5_000_000,
    )
    fitted_eval = evaluate_actor(
        cfg,
        actor,
        encoder,
        world_model,
        device,
        episodes=args.eval_episodes,
        seed=args.seed,
    )

    return_improvement = (
        fitted_eval["eval_return_mean"] - original_eval["eval_return_mean"]
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "critic_source": critic_key,
        "device": device,
        "collection_seed_start": args.seed,
        "collection_episodes": args.episodes,
        "visited_states": visited_states,
        "confident_labels": len(ys),
        "policy_q_samples": args.policy_q_samples,
        "policy_q_horizon": int(cfg.num_dream_steps),
        "label_hist": {
            str(action): int((ys == action).sum().detach().cpu())
            for action in range(cfg.n_actions)
        },
        "mean_abs_policy_q_delta": float(np.mean(np.abs(deltas))),
        "mean_policy_q_delta_se": float(np.mean(delta_ses)),
        "validation_fraction": args.validation_fraction,
        "train_labels": int(len(train_indices)),
        "validation_labels": int(len(validation_indices)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_seed_start": args.seed + 10_000,
        "original_eval": original_eval,
        "fitted_eval": fitted_eval,
        "return_improvement": float(return_improvement),
        "gate_pass": bool(
            fit_summary["validation_accuracy"] >= 0.8
            and return_improvement >= 50.0
        ),
        **fit_summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
