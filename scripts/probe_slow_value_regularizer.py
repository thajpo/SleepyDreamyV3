#!/usr/bin/env python3
"""Compare local and reference slow-value targets on fixed CartPole histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import infer_config_from_checkpoint, resolve_device
from dreamer.models import (
    initialize_critic,
    initialize_world_model,
    symlog,
    symexp_twohot_bins,
    twohot_encode,
    twohot_expectation,
    unimix_logits,
)


def slow_regularizer_metrics(
    online_logits: torch.Tensor,
    slow_logits: torch.Tensor,
    bins: torch.Tensor,
) -> dict[str, float | int]:
    """Summarize the two slow-target contracts and their logit gradients."""
    if online_logits.shape != slow_logits.shape:
        raise ValueError("online and slow logits must have identical shapes")
    if online_logits.ndim != 2 or online_logits.shape[-1] != bins.numel():
        raise ValueError("logits must have shape (states, bins)")

    online_probs = F.softmax(online_logits.float(), dim=-1)
    slow_probs = F.softmax(slow_logits.float(), dim=-1)
    slow_values = twohot_expectation(slow_logits.float(), bins.float())
    reference_targets = twohot_encode(slow_values, bins.float())

    eps = torch.finfo(online_probs.dtype).eps
    local_gradient = online_probs - slow_probs
    reference_gradient = online_probs - reference_targets
    local_norm = local_gradient.norm(dim=-1)
    reference_norm = reference_gradient.norm(dim=-1)
    valid_cosine = (local_norm > 1e-8) & (reference_norm > 1e-8)
    cosine = F.cosine_similarity(local_gradient, reference_gradient, dim=-1)

    slow_entropy = -(slow_probs * slow_probs.clamp_min(eps).log()).sum(dim=-1)
    reference_entropy = -(
        reference_targets * reference_targets.clamp_min(eps).log()
    ).sum(dim=-1)
    log_online = F.log_softmax(online_logits.float(), dim=-1)
    local_ce = -(slow_probs * log_online).sum(dim=-1)
    reference_ce = -(reference_targets * log_online).sum(dim=-1)
    total_variation = 0.5 * (slow_probs - reference_targets).abs().sum(dim=-1)

    metrics: dict[str, float | int] = {
        "states": int(online_logits.shape[0]),
        "mean_target_total_variation": float(total_variation.mean().cpu()),
        "median_target_total_variation": float(total_variation.median().cpu()),
        "mean_slow_target_entropy": float(slow_entropy.mean().cpu()),
        "mean_reference_target_entropy": float(reference_entropy.mean().cpu()),
        "mean_local_cross_entropy": float(local_ce.mean().cpu()),
        "mean_reference_cross_entropy": float(reference_ce.mean().cpu()),
        "mean_local_gradient_norm": float(local_norm.mean().cpu()),
        "mean_reference_gradient_norm": float(reference_norm.mean().cpu()),
        "local_near_zero_gradient_fraction": float(
            (local_norm <= 1e-8).float().mean().cpu()
        ),
        "valid_gradient_cosines": int(valid_cosine.sum().cpu()),
    }
    if bool(valid_cosine.any().item()):
        valid_values = cosine[valid_cosine]
        metrics.update(
            {
                "mean_gradient_cosine": float(valid_values.mean().cpu()),
                "median_gradient_cosine": float(valid_values.median().cpu()),
                "negative_gradient_cosine_fraction": float(
                    (valid_values < 0.0).float().mean().cpu()
                ),
            }
        )
    return metrics


def _initial_rssm_state(cfg, world_model, device: str):
    h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
    z = torch.zeros(
        1, world_model.n_latents, world_model.n_classes, device=device
    )
    return h, world_model.z_embedding(z.view(1, -1))


@torch.no_grad()
def run_probe(
    checkpoint_path: Path,
    *,
    device: str,
    states: int,
    seed: int,
) -> dict:
    cfg = infer_config_from_checkpoint(checkpoint_path, config_name=None)
    if cfg.environment_name != "CartPole-v1" or cfg.use_pixels:
        raise ValueError(f"{checkpoint_path} is not a state-only CartPole checkpoint")

    encoder, world_model = initialize_world_model(device, cfg, batch_size=1)
    critic = initialize_critic(device, cfg)
    slow_critic = initialize_critic(device, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "critic_ema" not in checkpoint:
        raise ValueError(f"{checkpoint_path} has no slow critic state")
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"], strict=False)
    critic.load_state_dict(checkpoint["critic"])
    slow_critic.load_state_dict(checkpoint["critic_ema"])
    for module in (encoder, world_model, critic, slow_critic):
        module.eval()

    env = gym.make(cfg.environment_name)
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    h, z_embed = _initial_rssm_state(cfg, world_model, device)
    online_batches: list[torch.Tensor] = []
    slow_batches: list[torch.Tensor] = []
    history = hashlib.sha256()
    episodes = 0
    states_collected = 0

    while states_collected < states:
        action = int(rng.integers(0, cfg.n_actions))
        observation, _reward, terminated, truncated, _ = env.step(action)
        state = np.asarray(observation, dtype=np.float32)
        action_onehot = F.one_hot(
            torch.tensor([action], device=device), num_classes=cfg.n_actions
        ).float()
        h, _prior_logits = world_model.step_dynamics(z_embed, action_onehot, h)
        tokens = encoder(symlog(torch.from_numpy(state).to(device).unsqueeze(0)))
        posterior_logits = unimix_logits(
            world_model.compute_posterior(h, tokens), unimix_ratio=0.01
        )
        posterior_index = posterior_logits.argmax(dim=-1)
        posterior = F.one_hot(
            posterior_index, num_classes=world_model.n_classes
        ).float()
        h_z = world_model.join_h_and_z(h, posterior)
        online_batches.append(critic(h_z).cpu())
        slow_batches.append(slow_critic(h_z).cpu())
        states_collected += 1
        history.update(state.tobytes())
        history.update(np.asarray([action], dtype=np.int64).tobytes())
        z_embed = world_model.z_embedding(posterior.view(1, -1))

        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset(seed=seed + episodes)
            h, z_embed = _initial_rssm_state(cfg, world_model, device)

    env.close()
    online_logits = torch.cat(online_batches)[:states]
    slow_logits = torch.cat(slow_batches)[:states]
    bins = symexp_twohot_bins(cfg.b_start, cfg.b_end, cfg.num_bins)
    metrics = slow_regularizer_metrics(online_logits, slow_logits, bins)
    return {
        "checkpoint": str(checkpoint_path),
        "train_step": checkpoint.get("train_step", checkpoint.get("step")),
        "device": device,
        "seed": seed,
        "episodes": episodes,
        "history_sha256": history.hexdigest(),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.states <= 0:
        parser.error("--states must be positive")

    device = resolve_device(args.device)
    summaries = []
    for checkpoint in args.checkpoints:
        print(f"Probing {checkpoint} on {device}...")
        summary = run_probe(
            checkpoint.resolve(), device=device, states=args.states, seed=args.seed
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
