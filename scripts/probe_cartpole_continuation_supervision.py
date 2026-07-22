#!/usr/bin/env python3
"""Test whether frozen CartPole latents make physical failure classifiable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import symlog
from dreamer.models.math_utils import unimix_logits

if __package__:
    from scripts.probe_cartpole_continuation import binary_roc_auc
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_continuation import binary_roc_auc  # type: ignore[import-not-found]
    from probe_cartpole_q import load_checkpoint_models  # type: ignore[import-not-found]


def build_failure_head(d_in: int, d_hidden: int) -> nn.Module:
    """Return the continuation head architecture with failure-logit output."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.RMSNorm(d_hidden),
        nn.SiLU(),
        nn.Linear(d_hidden, 1),
    )


def summarize_failure_scores(
    scores: torch.Tensor, terminals: torch.Tensor
) -> dict[str, float | int | None]:
    """Summarize failure probabilities at the fixed probability-0.5 cutoff."""
    scores_np = scores.detach().float().cpu().numpy()
    terminals_np = terminals.detach().bool().cpu().numpy()
    predicted = scores_np >= 0.5
    terminal_count = int(terminals_np.sum())
    live_count = int((~terminals_np).sum())
    terminal_recall = (
        float(predicted[terminals_np].mean()) if terminal_count else None
    )
    live_recall = (
        float((~predicted[~terminals_np]).mean()) if live_count else None
    )
    balanced_accuracy = (
        (terminal_recall + live_recall) / 2.0
        if terminal_recall is not None and live_recall is not None
        else None
    )
    return {
        "rows": int(scores_np.size),
        "terminal_rows": terminal_count,
        "live_rows": live_count,
        "failure_roc_auc": binary_roc_auc(
            scores_np.tolist(), terminals_np.tolist()
        ),
        "balanced_accuracy_at_half": balanced_accuracy,
        "terminal_recall_at_half": terminal_recall,
        "live_recall_at_half": live_recall,
        "terminal_failure_probability": (
            float(scores_np[terminals_np].mean()) if terminal_count else None
        ),
        "live_failure_probability": (
            float(scores_np[~terminals_np].mean()) if live_count else None
        ),
    }


@torch.no_grad()
def collect_dataset(cfg, actor, encoder, world_model, *, device: str, episodes: int, seed: int):
    """Collect post-action physical states and matching posterior-mode latents."""
    env = gym.make(cfg.environment_name)
    latents: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    terminals: list[bool] = []
    episode_ids: list[int] = []
    returns: list[float] = []
    truncated_episodes = 0

    def posterior(observation, h):
        state = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(device)
        tokens = encoder(symlog(state.unsqueeze(0)))
        logits = unimix_logits(
            world_model.compute_posterior(h, tokens), unimix_ratio=0.01
        )
        indices = logits.argmax(dim=-1)
        z = F.one_hot(indices, num_classes=world_model.n_classes).float()
        return world_model.join_h_and_z(h, z), world_model.z_embedding(z.view(1, -1))

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            z = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_embed = world_model.z_embedding(z.view(1, -1))
            zero_action = torch.zeros(1, cfg.n_actions, device=device)
            h, _ = world_model.step_dynamics(z_embed, zero_action, h)
            h_z, z_embed = posterior(obs, h)
            episode_return = 0.0

            while True:
                action = actor(h_z).argmax(dim=-1)
                action_onehot = F.one_hot(action, num_classes=cfg.n_actions).float()
                next_obs, reward, terminated, truncated, _ = env.step(
                    int(action.item())
                )
                episode_return += float(reward)
                h, _ = world_model.step_dynamics(z_embed, action_onehot, h)
                h_z, z_embed = posterior(next_obs, h)

                latents.append(h_z.squeeze(0).cpu())
                states.append(
                    torch.from_numpy(np.asarray(next_obs, dtype=np.float32)).cpu()
                )
                terminals.append(bool(terminated))
                episode_ids.append(episode)

                if terminated or truncated:
                    truncated_episodes += int(bool(truncated))
                    break
            returns.append(episode_return)
    finally:
        env.close()

    return {
        "latents": torch.stack(latents),
        "states": torch.stack(states),
        "terminals": torch.tensor(terminals, dtype=torch.bool),
        "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
        "returns": returns,
        "truncated_episodes": truncated_episodes,
    }


def fit_balanced_head(
    features: torch.Tensor,
    terminals: torch.Tensor,
    *,
    d_hidden: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[nn.Module, dict[str, float | int]]:
    """Fit one failure classifier with a frozen, preregistered class weight."""
    torch.manual_seed(seed)
    model = build_failure_head(features.shape[-1], d_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    x = features.to(device)
    y = terminals.float().to(device)
    terminal_count = int(y.sum().item())
    live_count = int(y.numel() - terminal_count)
    if terminal_count == 0 or live_count == 0:
        raise ValueError("training split must contain terminal and live rows")
    pos_weight = torch.tensor(live_count / terminal_count, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    final_loss = 0.0

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], generator=generator, device=device)
        for start in range(0, x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(x[indices]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, y[indices], pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    model.eval()
    return model, {
        "train_rows": int(y.numel()),
        "train_terminal_rows": terminal_count,
        "pos_weight": float(pos_weight.item()),
        "final_batch_loss": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if not 0 < args.train_episodes < args.episodes:
        parser.error("train-episodes must be between zero and episodes")

    device = resolve_device(args.device)
    (
        cfg,
        actor,
        _critic,
        _q_critic,
        encoder,
        world_model,
        checkpoint,
        _critic_key,
        _q_key,
    ) = load_checkpoint_models(args.checkpoint.resolve(), device)
    data = collect_dataset(
        cfg,
        actor,
        encoder,
        world_model,
        device=device,
        episodes=args.episodes,
        seed=args.seed,
    )
    train_mask = data["episode_ids"] < args.train_episodes
    test_mask = ~train_mask

    latent_head, latent_fit = fit_balanced_head(
        data["latents"][train_mask],
        data["terminals"][train_mask],
        d_hidden=cfg.d_hidden,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.fit_seed,
    )
    state_head, state_fit = fit_balanced_head(
        data["states"][train_mask],
        data["terminals"][train_mask],
        d_hidden=cfg.d_hidden,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.fit_seed,
    )

    with torch.no_grad():
        test_latents = data["latents"][test_mask].to(device)
        test_states = data["states"][test_mask].to(device)
        test_terminals = data["terminals"][test_mask]
        original_failure = 1.0 - torch.sigmoid(
            world_model.continue_predictor(test_latents).squeeze(-1)
        )
        latent_failure = torch.sigmoid(latent_head(test_latents).squeeze(-1))
        state_failure = torch.sigmoid(state_head(test_states).squeeze(-1))

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train_step": checkpoint.get("step", checkpoint.get("train_step")),
        "device": device,
        "episodes": args.episodes,
        "train_episodes": args.train_episodes,
        "seed_start": args.seed,
        "fit_seed": args.fit_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mean_episode_return": float(np.mean(data["returns"])),
        "truncated_episodes": data["truncated_episodes"],
        "latent_fit": latent_fit,
        "state_fit": state_fit,
        "original_head_test": summarize_failure_scores(
            original_failure, test_terminals
        ),
        "balanced_latent_head_test": summarize_failure_scores(
            latent_failure, test_terminals
        ),
        "balanced_true_state_head_test": summarize_failure_scores(
            state_failure, test_terminals
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
