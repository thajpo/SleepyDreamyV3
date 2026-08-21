#!/usr/bin/env python3
"""Cross solved/final Dreamer components against fixed CartPole real labels."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.config import Config
from dreamer.models import (
    learned_continue_discount,
    symexp_twohot_bins,
    twohot_expectation,
    unimix_logits,
)
from dreamer.models.dreaming import estimate_policy_lambda_action_values
from dreamer.trainer.replay_evidence import load_replay_evidence

if __package__:
    from scripts.probe_cartpole_checkpoint_drift import _observe_state, position_bin
    from scripts.probe_cartpole_q import (
        action_preference,
        load_checkpoint_models,
    )
    from scripts.probe_cartpole_recovery_value_boundary import (
        _inject_cartpole_state,
        _sample_one_step_prior_values,
        summarize_boundary_rows,
    )
    from scripts.probe_cartpole_replay_history_support import (
        _reset_latent,
        csv_fieldnames,
        select_actionable_rows,
        summarize_support_rows,
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
        _inject_cartpole_state,
        _sample_one_step_prior_values,
        summarize_boundary_rows,
    )
    from probe_cartpole_replay_history_support import (  # type: ignore[import-not-found]
        _reset_latent,
        csv_fieldnames,
        select_actionable_rows,
        summarize_support_rows,
    )


COMPONENT_NAMES = ("representation", "heads", "critic", "actor")
CHECKPOINT_NAMES = ("solved", "final")


@dataclass(frozen=True)
class LoadedModels:
    path: Path
    step: int
    cfg: Config
    actor: torch.nn.Module
    critic: torch.nn.Module
    encoder: torch.nn.Module
    world_model: torch.nn.Module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_cells() -> list[dict[str, str]]:
    """Return the complete deterministic 2^4 checkpoint-component matrix."""
    return [
        dict(zip(COMPONENT_NAMES, choices, strict=True))
        for choices in itertools.product(CHECKPOINT_NAMES, repeat=len(COMPONENT_NAMES))
    ]


def cell_name(components: dict[str, str]) -> str:
    return "__".join(f"{name}-{components[name]}" for name in COMPONENT_NAMES)


def load_fixed_label_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    """Load immutable real-policy branch labels keyed by replay occurrence."""
    required = {
        "sample_index",
        "t",
        "x",
        "x_dot",
        "theta",
        "theta_dot",
        "real_policy_score_0",
        "real_policy_score_1",
        "real_policy_pref",
    }
    labels: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"fixed-label CSV is missing fields: {sorted(missing)}")
        for row in reader:
            key = (int(row["sample_index"]), int(row["t"]))
            if key in labels:
                raise ValueError(f"duplicate fixed-label occurrence: {key}")
            labels[key] = row
    if not labels:
        raise ValueError("fixed-label CSV is empty")
    return labels


def validate_fixed_label_state(label: dict[str, str], state: np.ndarray) -> None:
    expected = np.asarray(
        [label["x"], label["x_dot"], label["theta"], label["theta_dot"]],
        dtype=np.float32,
    )
    if not np.allclose(expected, state, rtol=0.0, atol=1e-6):
        raise ValueError("fixed-label state does not match replay evidence")


def _config_signature(cfg) -> tuple:
    fields = (
        "environment_name",
        "use_pixels",
        "n_actions",
        "n_observations",
        "d_hidden",
        "num_latents",
        "rnn_n_blocks",
        "num_bins",
        "b_start",
        "b_end",
        "sequence_length",
        "replay_burn_in",
        "num_dream_steps",
        "gamma",
        "lam",
        "contdisc",
    )
    return tuple(getattr(cfg, field) for field in fields)


def load_models(path: Path, device: str) -> LoadedModels:
    (
        cfg,
        actor,
        critic,
        _q_critic,
        encoder,
        world_model,
        checkpoint,
        _critic_key,
        _q_key,
    ) = load_checkpoint_models(path, device, critic_source="online")
    step = int(checkpoint.get("step", checkpoint.get("train_step", -1)))
    return LoadedModels(path, step, cfg, actor, critic, encoder, world_model)


def validate_checkpoint_pair(solved: LoadedModels, final: LoadedModels) -> None:
    if _config_signature(solved.cfg) != _config_signature(final.cfg):
        raise ValueError("solved and final checkpoint structures differ")
    if solved.cfg.environment_name != "CartPole-v1" or solved.cfg.use_pixels:
        raise ValueError("component cross requires state-only CartPole checkpoints")
    if solved.step >= final.step:
        raise ValueError("solved checkpoint must precede final checkpoint")


def build_component_models(
    checkpoints: dict[str, LoadedModels], components: dict[str, str]
) -> tuple[Config, torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Build one cell while preserving each component's exact saved parameters."""
    representation = checkpoints[components["representation"]]
    heads = checkpoints[components["heads"]]
    critic_source = checkpoints[components["critic"]]
    actor_source = checkpoints[components["actor"]]

    encoder = copy.deepcopy(representation.encoder)
    world_model = copy.deepcopy(representation.world_model)
    world_model.reward_predictor = copy.deepcopy(heads.world_model.reward_predictor)
    world_model.continue_predictor = copy.deepcopy(heads.world_model.continue_predictor)
    critic = copy.deepcopy(critic_source.critic)
    actor = copy.deepcopy(actor_source.actor)
    for module in (encoder, world_model, critic, actor):
        module.eval()
    return representation.cfg, actor, critic, encoder, world_model


@torch.no_grad()
def posterior_one_step_q(
    env: gym.Env,
    state: np.ndarray,
    first_action: int,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    cfg,
    critic,
    encoder,
    world_model,
    bins: torch.Tensor,
) -> float:
    """Ground one candidate action in the simulator, then bootstrap the critic."""
    _inject_cartpole_state(env, state)
    next_obs, reward, terminated, truncated, _ = env.step(int(first_action))
    if terminated or truncated:
        return float(reward)
    previous_action = F.one_hot(
        torch.tensor([first_action], device=h.device), num_classes=cfg.n_actions
    ).float()
    _next_h, _next_z, next_h_z = _observe_state(
        cfg,
        encoder,
        world_model,
        h.clone(),
        z_embed.clone(),
        previous_action,
        np.asarray(next_obs, dtype=np.float32),
    )
    next_value = twohot_expectation(critic(next_h_z), bins)
    return float(reward) + float(cfg.gamma) * float(next_value.item())


@torch.no_grad()
def evaluate_component_cell(
    *,
    components: dict[str, str],
    checkpoints: dict[str, LoadedModels],
    evidence: dict,
    fixed_labels: dict[tuple[int, int], dict[str, str]],
    evidence_path: Path,
    fixed_labels_path: Path,
    out_dir: Path,
    device: str,
    model_samples: int,
    actionable_cap: int,
    seed: int,
    dream_batch_size: int,
) -> dict:
    cfg, actor, critic, encoder, world_model = build_component_models(
        checkpoints, components
    )
    if int(evidence["sequence_length"]) != int(cfg.sequence_length):
        raise ValueError("checkpoint and evidence sequence lengths differ")

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
    seen_label_keys: set[tuple[int, int]] = set()

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

                key = (sample_index, timestep)
                if key not in fixed_labels:
                    raise ValueError(f"fixed labels do not cover replay occurrence {key}")
                label = fixed_labels[key]
                validate_fixed_label_state(label, state)
                seen_label_keys.add(key)

                actor_logits = unimix_logits(
                    actor(h_z),
                    unimix_ratio=float(getattr(cfg, "actor_unimix", 0.01)),
                )
                actor_probabilities = F.softmax(actor_logits, dim=-1).squeeze(0)
                real_scores = [
                    float(label["real_policy_score_0"]),
                    float(label["real_policy_score_1"]),
                ]
                posterior_q = [
                    posterior_one_step_q(
                        branch_env,
                        state,
                        candidate,
                        h,
                        z_embed,
                        cfg,
                        critic,
                        encoder,
                        world_model,
                        bins,
                    )
                    for candidate in range(cfg.n_actions)
                ]
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
                        "real_policy_score_0": real_scores[0],
                        "real_policy_score_1": real_scores[1],
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

    if seen_label_keys != set(fixed_labels):
        unused = sorted(set(fixed_labels).difference(seen_label_keys))
        raise ValueError(f"fixed labels contain unused replay occurrences: {unused[:3]}")

    selected = select_actionable_rows(rows, cap=actionable_cap, seed=seed)
    prior_generator = torch.Generator(device=device).manual_seed(seed + 7_000_000)
    policy_generator = torch.Generator(device=device).manual_seed(seed + 8_000_000)
    for row in selected:
        h = row["_h"].to(device)
        z_embed = row["_z_embed"].to(device)
        h_z = row["_h_z"].to(device)
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
        prior_q = prior["q"].cpu().numpy()
        prior_q_se = prior["q_se"].cpu().numpy()
        prior_delta = float(prior_q[1] - prior_q[0])
        row.update(
            {
                "prior_critic_q0": float(prior_q[0]),
                "prior_critic_q1": float(prior_q[1]),
                "prior_critic_delta": prior_delta,
                "prior_critic_delta_se": float(
                    np.sqrt(prior_q_se[0] ** 2 + prior_q_se[1] ** 2)
                ),
                "prior_critic_pref": int(prior_delta > 0.0),
            }
        )
        for candidate in range(cfg.n_actions):
            row[f"prior_reward_{candidate}"] = float(prior["reward"][candidate])
            row[f"prior_continue_{candidate}"] = float(prior["continue"][candidate])
            row[f"prior_value_{candidate}"] = float(prior["value"][candidate])

    for start in range(0, len(selected), dream_batch_size):
        chunk = selected[start : start + dream_batch_size]
        h_z_batch = torch.cat([row["_h_z"].to(device) for row in chunk], dim=0)
        z_embed_batch = torch.cat(
            [row["_z_embed"].to(device) for row in chunk], dim=0
        )
        policy_q, policy_q_se = estimate_policy_lambda_action_values(
            h_z_batch,
            z_embed_batch,
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
        for index, row in enumerate(chunk):
            policy_values = policy_q[index].cpu().numpy()
            policy_se = policy_q_se[index].cpu().numpy()
            policy_delta = float(policy_values[1] - policy_values[0])
            row.update(
                {
                    "policy_q0": float(policy_values[0]),
                    "policy_q1": float(policy_values[1]),
                    "policy_q_delta": policy_delta,
                    "policy_q_delta_se": float(
                        np.sqrt(policy_se[0] ** 2 + policy_se[1] ** 2)
                    ),
                    "policy_q_pref": int(policy_delta > 0.0),
                }
            )

    support = summarize_support_rows(rows, selected)
    summary = {
        "cell": cell_name(components),
        "components": components,
        "evidence": str(evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "evidence_train_step": int(evidence["train_step"]),
        "fixed_labels": str(fixed_labels_path),
        "fixed_labels_sha256": file_sha256(fixed_labels_path),
        "fixed_real_policy_checkpoint": str(checkpoints["solved"].path),
        "fixed_real_policy_step": checkpoints["solved"].step,
        "selection_seed": seed,
        "model_samples": model_samples,
        "actionable_cap": actionable_cap,
        **support,
        "selected_boundary": summarize_boundary_rows(selected) if selected else None,
    }

    def serializable(row: dict) -> dict:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [serializable(row) for row in rows]
    with (out_dir / "all_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(all_rows))
        writer.writeheader()
        writer.writerows(all_rows)
    if selected:
        selected_rows = [serializable(row) for row in selected]
        with (out_dir / "selected_actionable_rows.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(selected_rows))
            writer.writeheader()
            writer.writerows(selected_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_component_cross(
    *,
    solved_checkpoint: Path,
    final_checkpoint: Path,
    evidence_path: Path,
    fixed_labels_path: Path,
    out_dir: Path,
    device: str,
    model_samples: int,
    actionable_cap: int,
    seed: int,
    dream_batch_size: int,
    max_cells: int | None = None,
) -> dict:
    solved = load_models(solved_checkpoint, device)
    final = load_models(final_checkpoint, device)
    validate_checkpoint_pair(solved, final)
    checkpoints = {"solved": solved, "final": final}
    evidence = load_replay_evidence(evidence_path)
    fixed_labels = load_fixed_label_rows(fixed_labels_path)

    summaries = []
    cells = component_cells()
    if max_cells is not None:
        cells = cells[:max_cells]
    for components in cells:
        name = cell_name(components)
        print(f"component-cross: {name}", flush=True)
        summaries.append(
            evaluate_component_cell(
                components=components,
                checkpoints=checkpoints,
                evidence=evidence,
                fixed_labels=fixed_labels,
                evidence_path=evidence_path,
                fixed_labels_path=fixed_labels_path,
                out_dir=out_dir / name,
                device=device,
                model_samples=model_samples,
                actionable_cap=actionable_cap,
                seed=seed,
                dream_batch_size=dream_batch_size,
            )
        )

    matrix = {
        "solved_checkpoint": str(solved_checkpoint),
        "solved_checkpoint_sha256": file_sha256(solved_checkpoint),
        "solved_step": solved.step,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "final_step": final.step,
        "evidence": str(evidence_path),
        "fixed_labels": str(fixed_labels_path),
        "complete_matrix": len(summaries) == len(component_cells()),
        "cells": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix_summary.json").write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solved-checkpoint", required=True, type=Path)
    parser.add_argument("--final-checkpoint", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--fixed-labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-samples", type=int, default=64)
    parser.add_argument("--actionable-cap", type=int, default=760)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dream-batch-size", type=int, default=32)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if (
        args.model_samples < 2
        or args.actionable_cap <= 0
        or args.cpu_threads <= 0
        or args.dream_batch_size <= 0
        or (args.max_cells is not None and args.max_cells <= 0)
    ):
        parser.error("samples, cap, threads, and optional max cells must be positive")

    device = resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)

    matrix = run_component_cross(
        solved_checkpoint=args.solved_checkpoint.resolve(),
        final_checkpoint=args.final_checkpoint.resolve(),
        evidence_path=args.evidence.resolve(),
        fixed_labels_path=args.fixed_labels.resolve(),
        out_dir=args.out.resolve(),
        device=device,
        model_samples=args.model_samples,
        actionable_cap=args.actionable_cap,
        seed=args.seed,
        dream_batch_size=args.dream_batch_size,
        max_cells=args.max_cells,
    )
    print(f"completed {len(matrix['cells'])} component cells")


if __name__ == "__main__":
    main()
