#!/usr/bin/env python3
"""Measure truncated replay carry against a complete CartPole prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.config import Config, validate_config
from dreamer.models import (
    initialize_actor,
    initialize_world_model,
    symlog,
    unimix_logits,
)
from dreamer.run_manifest import atomic_write_json, capture_git_state


def reference_config(seed: int) -> Config:
    """Return the exact size-1M state architecture used by the next canary."""
    config = Config(
        device="cpu",
        seed=int(seed),
        architecture_contract="reference_v3_state",
        d_hidden=64,
        num_latents=32,
        rnn_n_blocks=8,
        rssm_core="reference",
        continue_head_layers=1,
        vector_encoder_mode="reference",
        posterior_head_layers=1,
        replay_row_alignment="reference",
        replay_sequence_mode="stream",
        online_replay=True,
        continuous_replay_delivery=True,
        replay_burn_in=8,
    )
    validate_config(config)
    return config


def collect_random_episode(
    env: gym.Env,
    *,
    seed: int,
    n_actions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect reference-aligned observations and previous actions."""
    observation, _info = env.reset(seed=int(seed))
    action_rng = np.random.default_rng(int(seed) + 100_000)
    states = [np.asarray(observation, dtype=np.float32)]
    previous_actions = [np.zeros(n_actions, dtype=np.float32)]
    is_first = [True]
    while True:
        action = int(action_rng.integers(n_actions))
        observation, _reward, terminated, truncated, _info = env.step(action)
        states.append(np.asarray(observation, dtype=np.float32))
        previous_actions.append(np.eye(n_actions, dtype=np.float32)[action])
        is_first.append(False)
        if terminated or truncated:
            break
    return (
        torch.from_numpy(np.stack(states)),
        torch.from_numpy(np.stack(previous_actions)),
        torch.tensor(is_first, dtype=torch.bool),
    )


@torch.no_grad()
def deterministic_rollout(
    encoder,
    world_model,
    states: torch.Tensor,
    previous_actions: torch.Tensor,
    is_first: torch.Tensor,
) -> torch.Tensor:
    """Roll out posterior probabilities without categorical sampling."""
    tokens = encoder(symlog(states))
    batch_size = 1
    h = torch.zeros(batch_size, world_model.n_blocks * world_model.d_hidden)
    z = torch.zeros(batch_size, world_model.n_latents, world_model.n_classes)
    features = []
    for index in range(len(states)):
        action = previous_actions[index : index + 1]
        if bool(is_first[index].item()):
            h.zero_()
            z.zero_()
            action = torch.zeros_like(action)
        z_embed = world_model.z_embedding(z.reshape(batch_size, -1))
        h, _prior_logits = world_model.step_dynamics(z_embed, action, h)
        posterior_logits = world_model.compute_posterior(
            h, tokens[index : index + 1]
        )
        z = F.softmax(unimix_logits(posterior_logits, 0.01), dim=-1)
        features.append(world_model.join_h_and_z(h, z).squeeze(0))
    return torch.stack(features)


def metric_row(
    full_feature: torch.Tensor,
    replay_feature: torch.Tensor,
    actor,
    *,
    h_dim: int,
) -> dict[str, float]:
    difference = replay_feature - full_feature
    denominator = full_feature.norm().clamp_min(1e-8)
    cosine = F.cosine_similarity(
        full_feature.unsqueeze(0), replay_feature.unsqueeze(0)
    ).item()
    full_probs = F.softmax(unimix_logits(actor(full_feature.unsqueeze(0)), 0.01), -1)
    replay_probs = F.softmax(
        unimix_logits(actor(replay_feature.unsqueeze(0)), 0.01), -1
    )
    return {
        "feature_relative_l2": float((difference.norm() / denominator).item()),
        "feature_cosine": float(cosine),
        "h_relative_l2": float(
            (
                difference[:h_dim].norm()
                / full_feature[:h_dim].norm().clamp_min(1e-8)
            ).item()
        ),
        "z_mean_absolute_error": float(
            difference[h_dim:].abs().mean().item()
        ),
        "actor_probability_l1": float(
            (full_probs - replay_probs).abs().sum().item()
        ),
        "actor_action_agreement": float(
            full_probs.argmax(-1).item() == replay_probs.argmax(-1).item()
        ),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("carry parity requires at least one comparison row")
    result: dict[str, float | int] = {"comparisons": len(rows)}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_median"] = float(np.median(values))
        result[f"{key}_p95"] = float(np.quantile(values, 0.95))
        result[f"{key}_min"] = float(values.min())
    return result


def run_probe(
    *,
    seed: int,
    episode_seeds: list[int],
    burn_ins: list[int],
) -> dict:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    config = reference_config(seed)
    encoder, world_model = initialize_world_model("cpu", config, batch_size=1)
    actor = initialize_actor("cpu", config)
    encoder.eval()
    world_model.eval()
    actor.eval()

    rows_by_burn: dict[int, list[dict[str, float]]] = defaultdict(list)
    episode_lengths = []
    env = gym.make(config.environment_name)
    try:
        for episode_seed in episode_seeds:
            states, previous_actions, is_first = collect_random_episode(
                env, seed=episode_seed, n_actions=config.n_actions
            )
            episode_lengths.append(len(states) - 1)
            full = deterministic_rollout(
                encoder, world_model, states, previous_actions, is_first
            )
            for burn_in in burn_ins:
                for target in range(burn_in, len(states)):
                    start = target - burn_in
                    replay = deterministic_rollout(
                        encoder,
                        world_model,
                        states[start : target + 1],
                        previous_actions[start : target + 1],
                        is_first[start : target + 1],
                    )
                    rows_by_burn[burn_in].append(
                        metric_row(
                            full[target],
                            replay[-1],
                            actor,
                            h_dim=config.d_hidden * config.rnn_n_blocks,
                        )
                    )
    finally:
        env.close()

    summaries = {
        str(burn_in): summarize(rows_by_burn[burn_in]) for burn_in in burn_ins
    }
    primary = summaries["8"] if "8" in summaries else None
    primary_pass = None
    if primary is not None:
        primary_pass = bool(
            primary["actor_action_agreement_mean"] >= 0.99
            and primary["feature_cosine_median"] >= 0.99
            and primary["feature_relative_l2_p95"] <= 0.10
        )
    return {
        "schema_version": 1,
        "git": capture_git_state(),
        "model_seed": int(seed),
        "episode_seeds": episode_seeds,
        "episode_lengths": episode_lengths,
        "burn_ins": burn_ins,
        "deterministic_posterior": "unimixed_probabilities",
        "model_contract": {
            "architecture_contract": config.architecture_contract,
            "d_hidden": config.d_hidden,
            "deterministic_width": config.d_hidden * config.rnn_n_blocks,
            "stochastic_shape": [config.num_latents, config.d_hidden // 16],
            "replay_row_alignment": config.replay_row_alignment,
        },
        "summaries": summaries,
        "primary_burn_in": 8,
        "primary_thresholds": {
            "actor_action_agreement_mean_min": 0.99,
            "feature_cosine_median_min": 0.99,
            "feature_relative_l2_p95_max": 0.10,
        },
        "primary_pass": primary_pass,
        "limitation": "Initialized-model parity must be repeated after training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seed-start", type=int, default=17)
    parser.add_argument("--burn-ins", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    burn_ins = sorted(set(int(value) for value in args.burn_ins))
    if any(value <= 0 for value in burn_ins):
        raise ValueError("burn-ins must be positive")
    episode_seeds = list(
        range(args.episode_seed_start, args.episode_seed_start + args.episodes)
    )
    result = run_probe(
        seed=args.seed,
        episode_seeds=episode_seeds,
        burn_ins=burn_ins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(f"wrote {args.output}")
    for burn_in, summary in result["summaries"].items():
        print(
            f"burn_in={burn_in} n={summary['comparisons']} "
            f"cos_med={summary['feature_cosine_median']:.6f} "
            f"rel_l2_p95={summary['feature_relative_l2_p95']:.6f} "
            f"actor_agree={summary['actor_action_agreement_mean']:.6f}"
        )
    print(f"primary_pass={result['primary_pass']}")


if __name__ == "__main__":
    main()
