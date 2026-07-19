#!/usr/bin/env python3
"""Test whether frozen CartPole latents can fit trusted state values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.inspect import infer_config_from_checkpoint, resolve_device
from dreamer.models import (
    initialize_critic,
    initialize_world_model,
    symexp,
    symlog,
    twohot_encode,
    unimix_logits,
)
from dreamer.models.encoder import ThreeLayerMLP


def heuristic_action(state: np.ndarray) -> int:
    x, x_dot, theta, theta_dot = [float(v) for v in state]
    score = 0.8 * x + 1.0 * x_dot + 6.0 * theta + 1.0 * theta_dot
    return int(score > 0.0)


def load_world_model(checkpoint_path: Path, device: str):
    cfg = infer_config_from_checkpoint(checkpoint_path, config_name=None)
    if cfg.environment_name != "CartPole-v1" or cfg.use_pixels:
        raise ValueError(f"{checkpoint_path} is not a state-only CartPole checkpoint")
    encoder, world_model = initialize_world_model(device, cfg, batch_size=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"], strict=False)
    encoder.eval()
    world_model.eval()
    for module in (encoder, world_model):
        for param in module.parameters():
            param.requires_grad_(False)
    return cfg, encoder, world_model


@torch.no_grad()
def observe_latent(
    encoder,
    world_model,
    obs: np.ndarray,
    h: torch.Tensor,
    prev_action: torch.Tensor,
    z_embed: torch.Tensor,
):
    state = symlog(torch.from_numpy(obs).to(h.device).float().unsqueeze(0))
    h, _prior_logits = world_model.step_dynamics(z_embed, prev_action, h)
    posterior_logits = world_model.compute_posterior(h, encoder(state))
    posterior_logits = unimix_logits(posterior_logits, unimix_ratio=0.01)
    z_indices = posterior_logits.argmax(dim=-1)
    z_sample = F.one_hot(z_indices, num_classes=world_model.n_classes).float()
    h_z = world_model.join_h_and_z(h, z_sample)
    return h_z, h, world_model.z_embedding(z_sample.view(1, -1))


@torch.no_grad()
def trusted_remaining_return(
    env: gym.Env,
    state: np.ndarray,
    gamma: float,
    target_policy: str,
    max_steps: int = 500,
) -> float:
    """Return discounted survival under a deterministic target policy."""
    env.reset()
    env.unwrapped.state = np.asarray(state, dtype=np.float64).copy()
    env.unwrapped.steps_beyond_terminated = None
    total = 0.0
    discount = 1.0
    obs = np.asarray(state, dtype=np.float32)
    for _step in range(max_steps):
        if target_policy == "heuristic":
            action = heuristic_action(obs)
        elif target_policy == "action0":
            action = 0
        elif target_policy == "action1":
            action = 1
        else:
            raise ValueError(f"unsupported target policy: {target_policy!r}")
        obs, reward, terminated, truncated, _ = env.step(action)
        total += discount * float(reward)
        if terminated or truncated:
            break
        discount *= gamma
    return total


def episode_split(episode_ids: torch.Tensor, test_fraction: float, seed: int):
    """Split whole episodes so temporally adjacent states cannot cross splits."""
    unique = torch.unique(episode_ids).cpu().numpy()
    if len(unique) < 2:
        raise ValueError("critic supervision needs states from at least two episodes")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    test_episodes = max(1, int(round(len(unique) * test_fraction)))
    test_ids = torch.tensor(unique[:test_episodes], device=episode_ids.device)
    test_mask = torch.isin(episode_ids, test_ids)
    return (~test_mask).nonzero(as_tuple=False).squeeze(-1), test_mask.nonzero(
        as_tuple=False
    ).squeeze(-1)


def collect_dataset(
    cfg,
    encoder,
    world_model,
    device: str,
    states: int,
    seed: int,
    random_action_prob: float,
    target_policy: str,
):
    behavior_env = gym.make(cfg.environment_name)
    value_env = gym.make(cfg.environment_name)
    rng = np.random.default_rng(seed)
    obs, _ = behavior_env.reset(seed=seed)
    h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
    prev_action = torch.zeros(1, cfg.n_actions, device=device)
    z_prev = torch.zeros(
        1, world_model.n_latents, world_model.n_classes, device=device
    )
    z_embed = world_model.z_embedding(z_prev.view(1, -1))

    latents: list[torch.Tensor] = []
    true_states: list[torch.Tensor] = []
    targets: list[float] = []
    episode_ids: list[int] = []
    episode = 0
    while len(latents) < states:
        state = np.asarray(obs, dtype=np.float32)
        h_z, h, next_z_embed = observe_latent(
            encoder, world_model, state, h, prev_action, z_embed
        )
        latents.append(h_z.squeeze(0).detach().cpu())
        true_states.append(torch.from_numpy(state.copy()))
        targets.append(
            trusted_remaining_return(value_env, state, cfg.gamma, target_policy)
        )
        episode_ids.append(episode)

        trusted_action = heuristic_action(state)
        if rng.random() < random_action_prob:
            action = int(rng.integers(0, cfg.n_actions))
        else:
            action = trusted_action
        obs, _reward, terminated, truncated, _ = behavior_env.step(action)
        prev_action = F.one_hot(
            torch.tensor([action], device=device), num_classes=cfg.n_actions
        ).float()
        z_embed = next_z_embed.detach()

        if terminated or truncated:
            episode += 1
            obs, _ = behavior_env.reset(seed=seed + episode)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            prev_action = torch.zeros(1, cfg.n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_embed = world_model.z_embedding(z_prev.view(1, -1))

    behavior_env.close()
    value_env.close()
    return (
        torch.stack(latents).to(device),
        symlog(torch.stack(true_states).to(device)),
        torch.tensor(targets, device=device),
        torch.tensor(episode_ids, device=device),
    )


def initialize_state_critic(cfg, device: str) -> nn.Module:
    critic = ThreeLayerMLP(
        d_in=cfg.n_observations,
        d_hidden=cfg.d_hidden,
        d_out=cfg.num_bins,
    ).to(device)
    nn.init.zeros_(critic.mlp[-1].weight)
    nn.init.zeros_(critic.mlp[-1].bias)
    return critic


def pearson(predicted: torch.Tensor, target: torch.Tensor) -> float | None:
    pred = predicted.detach().float().cpu().numpy()
    true = target.detach().float().cpu().numpy()
    if pred.size < 2 or np.std(pred) < 1e-8 or np.std(true) < 1e-8:
        return None
    correlation = float(np.corrcoef(pred, true)[0, 1])
    return correlation if np.isfinite(correlation) else None


@torch.no_grad()
def evaluate(critic, xs, ys, bins, indices) -> dict:
    logits = critic(xs[indices])
    predicted = (F.softmax(logits, dim=-1) * bins).sum(dim=-1)
    target = ys[indices]
    error = predicted - target
    return {
        "samples": int(indices.numel()),
        "mae": float(error.abs().mean().cpu()),
        "rmse": float(error.square().mean().sqrt().cpu()),
        "pearson": pearson(predicted, target),
        "prediction_std": float(predicted.std(unbiased=False).cpu()),
        "target_std": float(target.std(unbiased=False).cpu()),
    }


def train_critic(
    critic,
    xs,
    ys,
    bins,
    train_indices,
    test_indices,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr)
    for _epoch in range(epochs):
        order = train_indices[
            torch.randperm(train_indices.numel(), device=train_indices.device)
        ]
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            logits = critic(xs[indices])
            targets = twohot_encode(ys[indices], bins)
            loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        "train": evaluate(critic, xs, ys, bins, train_indices),
        "test": evaluate(critic, xs, ys, bins, test_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random-action-prob", type=float, default=1.0)
    parser.add_argument(
        "--target-policy",
        choices=("action0", "action1", "heuristic"),
        default="action0",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    cfg, encoder, world_model = load_world_model(args.checkpoint.resolve(), device)
    latents, states, targets, episode_ids = collect_dataset(
        cfg,
        encoder,
        world_model,
        device,
        states=args.states,
        seed=args.seed,
        random_action_prob=args.random_action_prob,
        target_policy=args.target_policy,
    )
    train_indices, test_indices = episode_split(
        episode_ids, args.test_fraction, args.seed
    )
    bins = symexp(
        torch.linspace(cfg.b_start, cfg.b_end, cfg.num_bins, device=device)
    )

    latent_critic = initialize_critic(device, cfg)
    latent_result = train_critic(
        latent_critic,
        latents,
        targets,
        bins,
        train_indices,
        test_indices,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    torch.manual_seed(args.seed)
    state_critic = initialize_state_critic(cfg, device)
    state_result = train_critic(
        state_critic,
        states,
        targets,
        bins,
        train_indices,
        test_indices,
        args.epochs,
        args.batch_size,
        args.lr,
    )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device,
        "states": args.states,
        "episodes": int(torch.unique(episode_ids).numel()),
        "epochs": args.epochs,
        "seed": args.seed,
        "random_action_prob": args.random_action_prob,
        "target_policy": args.target_policy,
        "target_min": float(targets.min().cpu()),
        "target_max": float(targets.max().cpu()),
        "target_mean": float(targets.mean().cpu()),
        "posterior_latent_critic": latent_result,
        "true_state_critic": state_result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
