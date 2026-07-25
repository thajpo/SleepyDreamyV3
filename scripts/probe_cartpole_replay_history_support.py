#!/usr/bin/env python3
"""Test CartPole value ordering on checkpoint-matched replay histories."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import (
    learned_continue_discount,
    symexp_twohot_bins,
    unimix_logits,
)
from dreamer.models.dreaming import estimate_policy_lambda_action_values
from dreamer.trainer.replay_evidence import load_replay_evidence

if __package__:
    from scripts.probe_cartpole_checkpoint_drift import _observe_state, position_bin
    from scripts.probe_cartpole_q import action_preference, load_checkpoint_models
    from scripts.probe_cartpole_recovery_value_boundary import (
        _real_policy_branch,
        _sample_one_step_prior_values,
        summarize_boundary_rows,
    )
else:
    from probe_cartpole_checkpoint_drift import (  # type: ignore[import-not-found]
        _observe_state,
        position_bin,
    )
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        action_preference,
        load_checkpoint_models,
    )
    from probe_cartpole_recovery_value_boundary import (  # type: ignore[import-not-found]
        _real_policy_branch,
        _sample_one_step_prior_values,
        summarize_boundary_rows,
    )


def select_actionable_rows(
    rows: list[dict], *, cap: int, seed: int
) -> list[dict]:
    """Uniformly cap actionable replay occurrences without replacement."""
    if cap <= 0:
        raise ValueError("actionable cap must be positive")
    actionable = [row for row in rows if int(row["real_policy_pref"]) >= 0]
    if len(actionable) <= cap:
        return actionable
    return random.Random(seed).sample(actionable, cap)


def summarize_support_rows(rows: list[dict], selected: list[dict]) -> dict:
    """Describe replay actionability before the expensive dream-value cap."""
    actionable = [row for row in rows if int(row["real_policy_pref"]) >= 0]
    starts = {
        (
            int(row["start_collector_id"]),
            int(row["start_episode_id"]),
            int(row["start_offset"]),
        )
        for row in rows
    }
    return {
        "sampled_rows": len(rows),
        "sampled_actionable_rows": len(actionable),
        "sampled_actionable_fraction": len(actionable) / len(rows) if rows else None,
        "sampled_real_policy_pref_hist": dict(
            Counter(str(row["real_policy_pref"]) for row in rows)
        ),
        "selected_actionable_rows": len(selected),
        "selected_real_policy_pref_hist": dict(
            Counter(str(row["real_policy_pref"]) for row in selected)
        ),
        "sampled_sequence_starts": len(
            {int(row["sample_index"]) for row in rows}
        ),
        "unique_replay_sequence_starts": len(starts),
    }


def _reset_latent(cfg, world_model, device: str):
    h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
    z = torch.zeros(
        1,
        world_model.n_latents,
        world_model.n_classes,
        device=device,
    )
    return h, world_model.z_embedding(z.view(1, -1))


@torch.no_grad()
def run_replay_history_probe(
    checkpoint_path: Path,
    evidence_path: Path,
    out_dir: Path,
    *,
    device: str,
    real_horizon: int,
    model_samples: int,
    actionable_cap: int,
    seed: int,
) -> dict:
    """Evaluate one checkpoint on its own read-only replay evidence sample."""
    evidence = load_replay_evidence(evidence_path)
    (
        cfg,
        actor,
        critic,
        _q_critic,
        encoder,
        world_model,
        checkpoint,
        critic_key,
        _q_key,
    ) = load_checkpoint_models(checkpoint_path, device, critic_source="online")
    checkpoint_step = int(checkpoint.get("step", checkpoint.get("train_step", -1)))
    evidence_step = int(evidence["train_step"])
    if checkpoint_step != evidence_step:
        raise ValueError(
            f"checkpoint step {checkpoint_step} does not match evidence step "
            f"{evidence_step}"
        )
    if int(evidence["sequence_length"]) != int(cfg.sequence_length):
        raise ValueError("checkpoint and evidence sequence lengths differ")
    if cfg.environment_name != "CartPole-v1" or cfg.use_pixels:
        raise ValueError("replay-history support probe requires state-only CartPole")

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
    branch_env = gym.make(cfg.environment_name)
    rows: list[dict] = []
    world_model_h_backup = world_model.h_prev.clone()

    try:
        for sample_index in range(int(evidence["sample_count"])):
            h, z_embed = _reset_latent(cfg, world_model, device)
            for timestep in range(int(cfg.sequence_length)):
                if bool(evidence["is_first"][sample_index, timestep]):
                    h, z_embed = _reset_latent(cfg, world_model, device)
                state = evidence["states"][sample_index, timestep].astype(
                    np.float32, copy=False
                )
                previous_action = torch.from_numpy(
                    evidence["actions"][sample_index, timestep]
                ).to(device=device, dtype=torch.float32).unsqueeze(0)
                h, z_embed, h_z = _observe_state(
                    cfg,
                    encoder,
                    world_model,
                    h,
                    z_embed,
                    previous_action,
                    state,
                )
                if timestep < int(cfg.replay_burn_in):
                    continue

                actor_logits = unimix_logits(
                    actor(h_z),
                    unimix_ratio=float(getattr(cfg, "actor_unimix", 0.01)),
                )
                actor_probabilities = F.softmax(actor_logits, dim=-1).squeeze(0)
                real_scores = []
                posterior_q = []
                for candidate in range(cfg.n_actions):
                    score, q_value = _real_policy_branch(
                        branch_env,
                        state,
                        candidate,
                        h,
                        z_embed,
                        cfg,
                        actor,
                        critic,
                        encoder,
                        world_model,
                        bins,
                        horizon=real_horizon,
                    )
                    real_scores.append(score)
                    posterior_q.append(q_value)
                real_delta = float(real_scores[1] - real_scores[0])
                posterior_delta = float(posterior_q[1] - posterior_q[0])
                rows.append(
                    {
                        "sample_index": sample_index,
                        "start_collector_id": int(
                            evidence["start_collector_id"][sample_index]
                        ),
                        "start_episode_id": int(
                            evidence["start_episode_id"][sample_index]
                        ),
                        "start_offset": int(evidence["start_offset"][sample_index]),
                        "t": timestep,
                        "x": float(state[0]),
                        "x_dot": float(state[1]),
                        "theta": float(state[2]),
                        "theta_dot": float(state[3]),
                        "abs_x_bin": position_bin(float(state[0])),
                        "actor_action": int(actor_logits.argmax(dim=-1).item()),
                        "actor_probability_0": float(actor_probabilities[0].item()),
                        "actor_probability_1": float(actor_probabilities[1].item()),
                        "real_policy_score_0": float(real_scores[0]),
                        "real_policy_score_1": float(real_scores[1]),
                        "real_policy_delta": real_delta,
                        "real_policy_pref": action_preference(real_scores),
                        "posterior_critic_q0": float(posterior_q[0]),
                        "posterior_critic_q1": float(posterior_q[1]),
                        "posterior_critic_delta": posterior_delta,
                        "posterior_critic_pref": int(posterior_delta > 0.0),
                        "_h": h.detach().cpu(),
                        "_z_embed": z_embed.detach().cpu(),
                        "_h_z": h_z.detach().cpu(),
                    }
                )
    finally:
        branch_env.close()
        world_model.h_prev = world_model_h_backup

    selected = select_actionable_rows(rows, cap=actionable_cap, seed=seed)
    prior_generator = torch.Generator(device=device).manual_seed(seed + 7_000_000)
    policy_generator = torch.Generator(device=device).manual_seed(seed + 8_000_000)
    for row in selected:
        h = row["_h"].to(device)
        z_embed = row["_z_embed"].to(device)
        h_z = row["_h_z"].to(device)
        h_prev_backup = world_model.h_prev.clone()
        try:
            prior = _sample_one_step_prior_values(
                h,
                z_embed,
                critic,
                world_model,
                bins,
                n_actions=cfg.n_actions,
                imagination_discount=imagination_discount,
                samples=model_samples,
                generator=prior_generator,
            )
            policy_q, policy_q_se = estimate_policy_lambda_action_values(
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
                model_samples,
                generator=policy_generator,
                terminal_reward_penalty=float(
                    getattr(cfg, "terminal_reward_penalty", 0.0)
                ),
                actor_unimix=float(getattr(cfg, "actor_unimix", 0.01)),
            )
        finally:
            world_model.h_prev = h_prev_backup

        prior_q = prior["q"].cpu().numpy()
        prior_q_se = prior["q_se"].cpu().numpy()
        policy_values = policy_q.squeeze(0).cpu().numpy()
        policy_se = policy_q_se.squeeze(0).cpu().numpy()
        prior_delta = float(prior_q[1] - prior_q[0])
        policy_delta = float(policy_values[1] - policy_values[0])
        row.update(
            {
                "prior_critic_q0": float(prior_q[0]),
                "prior_critic_q1": float(prior_q[1]),
                "prior_critic_delta": prior_delta,
                "prior_critic_delta_se": float(
                    np.sqrt(prior_q_se[0] ** 2 + prior_q_se[1] ** 2)
                ),
                "prior_critic_pref": int(prior_delta > 0.0),
                "policy_q0": float(policy_values[0]),
                "policy_q1": float(policy_values[1]),
                "policy_q_delta": policy_delta,
                "policy_q_delta_se": float(
                    np.sqrt(policy_se[0] ** 2 + policy_se[1] ** 2)
                ),
                "policy_q_pref": int(policy_delta > 0.0),
            }
        )
        for candidate in range(cfg.n_actions):
            row[f"prior_reward_{candidate}"] = float(
                prior["reward"][candidate].item()
            )
            row[f"prior_continue_{candidate}"] = float(
                prior["continue"][candidate].item()
            )
            row[f"prior_value_{candidate}"] = float(
                prior["value"][candidate].item()
            )

    support_summary = summarize_support_rows(rows, selected)
    summary = {
        "checkpoint": str(checkpoint_path),
        "evidence": str(evidence_path),
        "train_step": checkpoint_step,
        "checkpoint_name": str(evidence["checkpoint_name"]),
        "critic_used": critic_key,
        "selector": str(evidence["selector"]),
        "evidence_seed": int(evidence["seed"]),
        "selection_seed": seed,
        "real_horizon": real_horizon,
        "model_samples": model_samples,
        "actionable_cap": actionable_cap,
        **support_summary,
        "selected_boundary": (
            summarize_boundary_rows(selected) if selected else None
        ),
    }

    def serializable(row: dict) -> dict:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    out_dir.mkdir(parents=True, exist_ok=True)
    all_serialized = [serializable(row) for row in rows]
    with (out_dir / "all_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_serialized[0]))
        writer.writeheader()
        writer.writerows(all_serialized)
    if selected:
        selected_serialized = [serializable(row) for row in selected]
        with (out_dir / "selected_actionable_rows.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected_serialized[0]))
            writer.writeheader()
            writer.writerows(selected_serialized)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real-horizon", type=int, default=30)
    parser.add_argument("--model-samples", type=int, default=64)
    parser.add_argument("--actionable-cap", type=int, default=760)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.real_horizon <= 0 or args.model_samples < 2 or args.actionable_cap <= 0:
        parser.error("horizon/cap must be positive and model samples at least two")

    summary = run_replay_history_probe(
        args.checkpoint.resolve(),
        args.evidence.resolve(),
        args.out.resolve(),
        device=resolve_device(args.device),
        real_horizon=args.real_horizon,
        model_samples=args.model_samples,
        actionable_cap=args.actionable_cap,
        seed=args.seed,
    )
    boundary = summary["selected_boundary"] or {}
    print(
        f"step={summary['train_step']} rows={summary['sampled_rows']} "
        f"actionable={summary['sampled_actionable_rows']} "
        f"posterior={boundary.get('posterior_critic_vs_real_policy_balanced_accuracy')} "
        f"dream={boundary.get('policy_q_vs_real_policy_balanced_accuracy')} "
        f"actor={boundary.get('actor_vs_real_policy_balanced_accuracy')}"
    )


if __name__ == "__main__":
    main()
