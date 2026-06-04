#!/usr/bin/env python3
"""Test whether a frozen CartPole world model exposes enough state for control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import infer_config_from_checkpoint, resolve_device
from dreamer.models import (
    initialize_actor,
    initialize_world_model,
    symlog,
    unimix_logits,
)


def heuristic_action(state: np.ndarray) -> int:
    x, x_dot, theta, theta_dot = [float(v) for v in state]
    score = 0.8 * x + 1.0 * x_dot + 6.0 * theta + 1.0 * theta_dot
    return int(score > 0.0)


def load_world_model(checkpoint_path: Path, device: str, fresh_actor: bool):
    cfg = infer_config_from_checkpoint(checkpoint_path, config_name=None)
    if cfg.environment_name != "CartPole-v1" or cfg.use_pixels:
        raise ValueError(f"{checkpoint_path} is not a state-only CartPole checkpoint")
    actor = initialize_actor(device, cfg)
    encoder, world_model = initialize_world_model(device, cfg, batch_size=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not fresh_actor:
        actor.load_state_dict(checkpoint["actor"])
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"], strict=False)
    actor.train()
    encoder.eval()
    world_model.eval()
    for module in (encoder, world_model):
        for param in module.parameters():
            param.requires_grad_(False)
    return cfg, actor, encoder, world_model


@torch.no_grad()
def observe_latent(
    encoder,
    world_model,
    cfg,
    obs: np.ndarray,
    h: torch.Tensor,
    prev_action_onehot: torch.Tensor,
    z_prev_embed: torch.Tensor,
):
    state_vec = symlog(torch.from_numpy(obs).to(h.device).float().unsqueeze(0))
    h, _prior_logits = world_model.step_dynamics(z_prev_embed, prev_action_onehot, h)
    tokens = encoder(state_vec)
    posterior_logits = world_model.compute_posterior(h, tokens)
    posterior_logits = unimix_logits(posterior_logits, unimix_ratio=0.01)
    z_idx = posterior_logits.argmax(dim=-1)
    z_sample = F.one_hot(z_idx, num_classes=world_model.n_classes).float()
    h_z = world_model.join_h_and_z(h, z_sample)
    z_embed = world_model.z_embedding(z_sample.view(1, -1))
    return h_z, h, z_embed


def collect_dataset(
    cfg,
    encoder,
    world_model,
    device: str,
    states: int,
    seed: int,
    random_action_prob: float,
):
    env = gym.make(cfg.environment_name)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
    prev_action_onehot = torch.zeros(1, cfg.n_actions, device=device)
    z_prev = torch.zeros(
        1, world_model.n_latents, world_model.n_classes, device=device
    )
    z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))

    xs = []
    ys = []
    episode = 0
    while len(xs) < states:
        state = np.asarray(obs, dtype=np.float32)
        h_z, h, z_embed = observe_latent(
            encoder, world_model, cfg, state, h, prev_action_onehot, z_prev_embed
        )
        label = heuristic_action(state)
        xs.append(h_z.squeeze(0).detach().cpu())
        ys.append(label)

        if rng.random() < random_action_prob:
            action = int(rng.integers(0, cfg.n_actions))
        else:
            action = label
        obs, _reward, terminated, truncated, _ = env.step(action)
        prev_action_onehot = F.one_hot(
            torch.tensor([action], device=device), num_classes=cfg.n_actions
        ).float()
        z_prev_embed = z_embed.detach()

        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            prev_action_onehot = torch.zeros(1, cfg.n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))

    env.close()
    return torch.stack(xs).to(device), torch.tensor(ys, device=device)


def train_actor(actor, xs, ys, epochs: int, batch_size: int, lr: float) -> dict:
    opt = torch.optim.AdamW(actor.parameters(), lr=lr)
    n = xs.shape[0]
    losses = []
    accs = []
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=xs.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = actor(xs[idx])
            loss = F.cross_entropy(logits, ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = actor(xs).argmax(dim=-1)
            accs.append(float((pred == ys).float().mean().cpu()))
    return {"final_loss": losses[-1], "final_train_accuracy": accs[-1]}


@torch.no_grad()
def evaluate_actor(cfg, actor, encoder, world_model, device: str, episodes: int, seed: int):
    env = gym.make(cfg.environment_name)
    returns = []
    lengths = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10000 + ep)
        h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
        prev_action_onehot = torch.zeros(1, cfg.n_actions, device=device)
        z_prev = torch.zeros(
            1, world_model.n_latents, world_model.n_classes, device=device
        )
        z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
        total = 0.0
        steps = 0
        while True:
            state = np.asarray(obs, dtype=np.float32)
            h_z, h, z_embed = observe_latent(
                encoder, world_model, cfg, state, h, prev_action_onehot, z_prev_embed
            )
            action = int(actor(h_z).argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            prev_action_onehot = F.one_hot(
                torch.tensor([action], device=device), num_classes=cfg.n_actions
            ).float()
            z_prev_embed = z_embed.detach()
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
    env.close()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_returns": returns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/control_ablation/supervised_actor_probe"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random-action-prob", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fresh-actor", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    cfg, actor, encoder, world_model = load_world_model(
        args.checkpoint.resolve(), device, fresh_actor=args.fresh_actor
    )
    xs, ys = collect_dataset(
        cfg,
        encoder,
        world_model,
        device,
        states=args.states,
        seed=args.seed,
        random_action_prob=args.random_action_prob,
    )
    train_summary = train_actor(actor, xs, ys, args.epochs, args.batch_size, args.lr)
    eval_summary = evaluate_actor(
        cfg,
        actor,
        encoder,
        world_model,
        device,
        episodes=args.eval_episodes,
        seed=args.seed,
    )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device,
        "states": args.states,
        "epochs": args.epochs,
        "fresh_actor": bool(args.fresh_actor),
        "random_action_prob": args.random_action_prob,
        "label_hist": {
            str(i): int((ys == i).sum().detach().cpu()) for i in range(cfg.n_actions)
        },
        **train_summary,
        **eval_summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
