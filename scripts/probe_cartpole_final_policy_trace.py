#!/usr/bin/env python3
"""Trace the final CartPole policy on histories the solved policy made safe."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import symexp, symexp_twohot_bins, twohot_expectation, unimix_logits
from dreamer.trainer.replay_evidence import load_replay_evidence

if __package__:
    from scripts.probe_cartpole_checkpoint_drift import _observe_state
    from scripts.probe_cartpole_component_cross import (
        LoadedModels,
        csv_fieldnames,
        load_fixed_label_rows,
        load_models,
        validate_checkpoint_pair,
        validate_fixed_label_state,
    )
    from scripts.probe_cartpole_recovery_value_boundary import _inject_cartpole_state
    from scripts.probe_cartpole_replay_history_support import _reset_latent
else:
    from probe_cartpole_checkpoint_drift import _observe_state  # type: ignore[import-not-found]
    from probe_cartpole_component_cross import (  # type: ignore[import-not-found]
        LoadedModels,
        csv_fieldnames,
        load_fixed_label_rows,
        load_models,
        validate_checkpoint_pair,
        validate_fixed_label_state,
    )
    from probe_cartpole_recovery_value_boundary import (  # type: ignore[import-not-found]
        _inject_cartpole_state,
    )
    from probe_cartpole_replay_history_support import (  # type: ignore[import-not-found]
        _reset_latent,
    )


ACTOR_CROSS_KEYS = ("solved_solved", "final_solved", "solved_final", "final_final")


def select_final_only_keys(
    solved_labels: dict[tuple[int, int], dict[str, str]],
    final_labels: dict[tuple[int, int], dict[str, str]],
    *,
    cap: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Select occurrences tied under solved but actionable under final continuation."""
    if solved_labels.keys() != final_labels.keys():
        raise ValueError("solved and final label cohorts differ")
    eligible = sorted(
        key
        for key in solved_labels
        if int(solved_labels[key]["real_policy_pref"]) < 0
        and int(final_labels[key]["real_policy_pref"]) >= 0
    )
    if len(eligible) < cap:
        raise ValueError(f"final-only cohort has {len(eligible)} rows, below cap {cap}")
    return sorted(random.Random(seed).sample(eligible, cap))


def _mean(rows: list[dict], key: str) -> float | None:
    return float(np.mean([float(row[key]) for row in rows])) if rows else None


def summarize_actor_cross(rows: list[dict]) -> dict:
    """Describe which actor/representation swap transfers deployed action drift."""
    decision_rows = [row for row in rows if "solved_solved" in row]
    changed = [
        row
        for row in decision_rows
        if int(row["solved_solved"]) != int(row["final_final"])
    ]

    def agreement(left: str, right: str, cohort: list[dict]) -> float | None:
        if not cohort:
            return None
        return float(np.mean([int(row[left]) == int(row[right]) for row in cohort]))

    first_divergence = {}
    for row in decision_rows:
        branch = (int(row["sample_index"]), int(row["t"]), int(row["first_action"]))
        if int(row["solved_solved"]) != int(row["final_final"]):
            first_divergence[branch] = min(
                int(row["next_depth"]),
                first_divergence.get(branch, int(row["next_depth"])),
            )

    return {
        "decision_rows": len(decision_rows),
        "solved_final_changed_rows": len(changed),
        "action_histograms": {
            key: dict(Counter(str(row[key]) for row in decision_rows))
            for key in ACTOR_CROSS_KEYS
        },
        "solved_vs_final_agreement": agreement(
            "solved_solved", "final_final", decision_rows
        ),
        "actor_swap_on_solved_representation_agreement_with_final": agreement(
            "final_solved", "final_final", decision_rows
        ),
        "representation_swap_with_solved_actor_agreement_with_final": agreement(
            "solved_final", "final_final", decision_rows
        ),
        "changed_rows_actor_swap_transfers_final": agreement(
            "final_solved", "final_final", changed
        ),
        "changed_rows_representation_swap_transfers_final": agreement(
            "solved_final", "final_final", changed
        ),
        "branches_with_divergence": len(first_divergence),
        "first_divergence_depth_mean": (
            float(np.mean(list(first_divergence.values()))) if first_divergence else None
        ),
        "first_divergence_depth_histogram": dict(
            Counter(str(value) for value in first_divergence.values())
        ),
    }


@torch.no_grad()
def sample_prior_outcome(
    models: LoadedModels,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    action: int,
    bins: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample the final prior before the real successor observation is available."""
    cfg = cast(Any, models.cfg)
    world_model = cast(Any, models.world_model)
    action_ids = torch.full((samples,), int(action), device=h.device)
    action_onehot = F.one_hot(action_ids, num_classes=cfg.n_actions).float()
    h_next, prior_logits = world_model.step_dynamics(
        z_embed.expand(samples, -1), action_onehot, h.expand(samples, -1)
    )
    prior_probs = F.softmax(unimix_logits(prior_logits, unimix_ratio=0.01), dim=-1)
    flat = prior_probs.reshape(-1, prior_probs.shape[-1])
    indices = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    indices = indices.view(*prior_probs.shape[:-1])
    z_state = F.one_hot(indices, num_classes=world_model.n_classes).float()
    h_z = world_model.join_h_and_z(h_next, z_state)
    decoded_state = symexp(world_model.decoder(h_z)["state"])
    reward = twohot_expectation(world_model.reward_predictor(h_z), bins)
    continuation = torch.sigmoid(world_model.continue_predictor(h_z).squeeze(-1))
    return {
        "decoded_state_mean": decoded_state.mean(dim=0),
        "decoded_state_sample_mse_centered": (
            (decoded_state - decoded_state.mean(dim=0)) ** 2
        ).mean(),
        "reward_mean": reward.mean(),
        "continuation_mean": continuation.mean(),
    }


def _actor_cross(
    solved: LoadedModels,
    final: LoadedModels,
    solved_h_z: torch.Tensor,
    final_h_z: torch.Tensor,
) -> dict[str, int]:
    return {
        "solved_solved": int(solved.actor(solved_h_z).argmax(dim=-1).item()),
        "final_solved": int(final.actor(solved_h_z).argmax(dim=-1).item()),
        "solved_final": int(solved.actor(final_h_z).argmax(dim=-1).item()),
        "final_final": int(final.actor(final_h_z).argmax(dim=-1).item()),
    }


@torch.no_grad()
def reconstruct_selected_histories(
    evidence: dict,
    keys: list[tuple[int, int]],
    solved: LoadedModels,
    final: LoadedModels,
    final_labels: dict[tuple[int, int], dict[str, str]],
    *,
    device: str,
) -> list[dict]:
    selected = set(keys)
    histories = []
    for sample_index in sorted({key[0] for key in keys}):
        solved_h, solved_z = _reset_latent(solved.cfg, solved.world_model, device)
        final_h, final_z = _reset_latent(final.cfg, final.world_model, device)
        for timestep in range(int(solved.cfg.sequence_length)):
            if bool(evidence["is_first"][sample_index, timestep]):
                solved_h, solved_z = _reset_latent(
                    solved.cfg, solved.world_model, device
                )
                final_h, final_z = _reset_latent(final.cfg, final.world_model, device)
            state = evidence["states"][sample_index, timestep].astype(
                np.float32, copy=False
            )
            previous_action = torch.from_numpy(
                evidence["actions"][sample_index, timestep]
            ).to(device=device, dtype=torch.float32).unsqueeze(0)
            solved_h, solved_z, solved_h_z = _observe_state(
                solved.cfg,
                solved.encoder,
                solved.world_model,
                solved_h,
                solved_z,
                previous_action,
                state,
            )
            final_h, final_z, final_h_z = _observe_state(
                final.cfg,
                final.encoder,
                final.world_model,
                final_h,
                final_z,
                previous_action,
                state,
            )
            key = (sample_index, timestep)
            if key not in selected:
                continue
            validate_fixed_label_state(final_labels[key], state)
            histories.append(
                {
                    "sample_index": sample_index,
                    "t": timestep,
                    "state": state.copy(),
                    "solved_h": solved_h.clone(),
                    "solved_z": solved_z.clone(),
                    "solved_h_z": solved_h_z.clone(),
                    "final_h": final_h.clone(),
                    "final_z": final_z.clone(),
                    "final_h_z": final_h_z.clone(),
                }
            )
    if {(row["sample_index"], row["t"]) for row in histories} != selected:
        raise ValueError("failed to reconstruct every selected replay occurrence")
    return histories


@torch.no_grad()
def run_trace(
    *,
    solved_checkpoint: Path,
    final_checkpoint: Path,
    evidence_path: Path,
    solved_labels_path: Path,
    final_labels_path: Path,
    out_dir: Path,
    device: str,
    cap: int,
    selection_seed: int,
    model_samples: int,
    horizon: int,
) -> dict:
    solved = load_models(solved_checkpoint, device)
    final = load_models(final_checkpoint, device)
    validate_checkpoint_pair(solved, final)
    evidence = load_replay_evidence(evidence_path)
    solved_labels = load_fixed_label_rows(solved_labels_path)
    final_labels = load_fixed_label_rows(final_labels_path)
    keys = select_final_only_keys(
        solved_labels, final_labels, cap=cap, seed=selection_seed
    )
    histories = reconstruct_selected_histories(
        evidence, keys, solved, final, final_labels, device=device
    )
    bins = symexp_twohot_bins(
        final.cfg.b_start,
        final.cfg.b_end,
        int(final.cfg.num_bins),
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device).manual_seed(selection_seed + 9_000_000)
    env = gym.make(final.cfg.environment_name)
    rows: list[dict] = []
    score_checks = []

    try:
        for history in histories:
            key = (int(history["sample_index"]), int(history["t"]))
            label = final_labels[key]
            for first_action in range(final.cfg.n_actions):
                _inject_cartpole_state(env, history["state"])
                solved_h = history["solved_h"].clone()
                solved_z = history["solved_z"].clone()
                final_h = history["final_h"].clone()
                final_z = history["final_z"].clone()
                final_h_z = history["final_h_z"].clone()
                action = int(first_action)
                score = 0.0

                for depth in range(horizon):
                    predicted = sample_prior_outcome(
                        final,
                        final_h,
                        final_z,
                        action,
                        bins,
                        samples=model_samples,
                        generator=generator,
                    )
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = np.asarray(next_obs, dtype=np.float32)
                    score += float(reward)
                    done = bool(terminated or truncated)
                    predicted_state = predicted["decoded_state_mean"].cpu().numpy()
                    row = {
                        "sample_index": key[0],
                        "t": key[1],
                        "first_action": first_action,
                        "depth": depth,
                        "actual_action": action,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "predicted_reward": float(predicted["reward_mean"]),
                        "predicted_continuation": float(
                            predicted["continuation_mean"]
                        ),
                        "predicted_next_state_mse": float(
                            np.mean((predicted_state - next_state) ** 2)
                        ),
                        "predicted_state_sample_variance": float(
                            predicted["decoded_state_sample_mse_centered"]
                        ),
                    }
                    if not done:
                        previous_action = F.one_hot(
                            torch.tensor([action], device=device),
                            num_classes=final.cfg.n_actions,
                        ).float()
                        solved_h, solved_z, solved_h_z = _observe_state(
                            solved.cfg,
                            solved.encoder,
                            solved.world_model,
                            solved_h,
                            solved_z,
                            previous_action,
                            next_state,
                        )
                        final_h, final_z, final_h_z = _observe_state(
                            final.cfg,
                            final.encoder,
                            final.world_model,
                            final_h,
                            final_z,
                            previous_action,
                            next_state,
                        )
                        cross = _actor_cross(
                            solved, final, solved_h_z, final_h_z
                        )
                        row.update(cross)
                        row["next_depth"] = depth + 1
                        action = cross["final_final"]
                    rows.append(row)
                    if done:
                        break

                expected = float(label[f"real_policy_score_{first_action}"])
                score_checks.append(abs(score - expected))
                if abs(score - expected) > 1e-9:
                    raise ValueError(
                        f"real branch score {score} does not match retained label {expected}"
                    )
    finally:
        env.close()

    terminal_rows = [row for row in rows if row["terminated"]]
    live_rows = [row for row in rows if not row["terminated"]]
    summary = {
        "solved_checkpoint": str(solved_checkpoint),
        "solved_step": solved.step,
        "final_checkpoint": str(final_checkpoint),
        "final_step": final.step,
        "evidence": str(evidence_path),
        "solved_labels": str(solved_labels_path),
        "final_labels": str(final_labels_path),
        "selection_seed": selection_seed,
        "selected_histories": len(histories),
        "selected_keys": [list(key) for key in keys],
        "branches": len(score_checks),
        "horizon": horizon,
        "model_samples": model_samples,
        "branch_score_max_error": max(score_checks, default=None),
        "transitions": len(rows),
        "terminal_transitions": len(terminal_rows),
        "predicted_continuation_terminal_mean": _mean(
            terminal_rows, "predicted_continuation"
        ),
        "predicted_continuation_nonterminal_mean": _mean(
            live_rows, "predicted_continuation"
        ),
        "predicted_next_state_mse_mean": _mean(
            rows, "predicted_next_state_mse"
        ),
        "predicted_next_state_mse_terminal_mean": _mean(
            terminal_rows, "predicted_next_state_mse"
        ),
        "actor_cross": summarize_actor_cross(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solved-checkpoint", required=True, type=Path)
    parser.add_argument("--final-checkpoint", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--solved-labels", required=True, type=Path)
    parser.add_argument("--final-labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=23)
    parser.add_argument("--model-samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.cap, args.model_samples, args.horizon, args.cpu_threads) <= 0:
        parser.error("cap, samples, horizon, and threads must be positive")
    if args.model_samples < 2:
        parser.error("model samples must be at least two")
    device = resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    summary = run_trace(
        solved_checkpoint=args.solved_checkpoint.resolve(),
        final_checkpoint=args.final_checkpoint.resolve(),
        evidence_path=args.evidence.resolve(),
        solved_labels_path=args.solved_labels.resolve(),
        final_labels_path=args.final_labels.resolve(),
        out_dir=args.out.resolve(),
        device=device,
        cap=args.cap,
        selection_seed=args.selection_seed,
        model_samples=args.model_samples,
        horizon=args.horizon,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
