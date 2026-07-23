#!/usr/bin/env python3
"""Measure whether the pinned DreamerV3 RMSNorm epsilon matters in checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import symlog, symexp_twohot_bins, twohot_expectation

if __package__:
    from scripts.probe_cartpole_checkpoint_drift import _observe_state
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_checkpoint_drift import (  # type: ignore[import-not-found]
        _observe_state,
    )
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        load_checkpoint_models,
    )


LOGGER = logging.getLogger(__name__)
REFERENCE_EPSILON = 1e-4


def distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    """Return stable scalar summaries for a possibly empty measurement list."""
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def rmsnorm_input_effect(
    inputs: torch.Tensor, target_epsilon: float = REFERENCE_EPSILON
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row mean-square and relative scale change for default -> target eps."""
    values = inputs.detach().float()
    mean_square = values.square().mean(dim=-1).reshape(-1)
    default_epsilon = torch.finfo(inputs.dtype).eps
    target_over_default = torch.sqrt(
        (mean_square + default_epsilon) / (mean_square + target_epsilon)
    )
    return mean_square, (target_over_default - 1.0).abs()


def set_reference_epsilon(
    module: torch.nn.Module, epsilon: float = REFERENCE_EPSILON
) -> list[str]:
    """Change only RMSNorms that currently request PyTorch's dtype default."""
    changed = []
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.RMSNorm) and child.eps is None:
            child.eps = float(epsilon)
            changed.append(name)
    return changed


class RMSNormInputRecorder:
    """Record inputs and their same-input epsilon sensitivity via forward hooks."""

    def __init__(self, module: torch.nn.Module):
        self.mean_square: dict[str, list[float]] = {}
        self.relative_scale_change: dict[str, list[float]] = {}
        self._handles: list[Any] = []
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.RMSNorm) and child.eps is None:
                self.mean_square[name] = []
                self.relative_scale_change[name] = []
                self._handles.append(child.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def record(_module, args) -> None:
            mean_square, relative_change = rmsnorm_input_effect(args[0])
            self.mean_square[name].extend(mean_square.cpu().tolist())
            self.relative_scale_change[name].extend(relative_change.cpu().tolist())

        return record

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> dict[str, dict]:
        result = {}
        for name in self.mean_square:
            mean_square = self.mean_square[name]
            result[name] = {
                "input_mean_square": distribution_summary(mean_square),
                "fraction_input_mean_square_le_1e-4": (
                    float(np.mean(np.asarray(mean_square) <= REFERENCE_EPSILON))
                    if mean_square
                    else None
                ),
                "relative_output_scale_change": distribution_summary(
                    self.relative_scale_change[name]
                ),
            }
        return result


@torch.no_grad()
def _observe_target(cfg, encoder, world_model, h, z_embed, previous_action, state):
    h, prior_logits = world_model.step_dynamics(z_embed, previous_action, h)
    tokens = encoder(
        symlog(torch.from_numpy(state).to(h.device).float().unsqueeze(0))
    )
    posterior_logits = world_model.compute_posterior(h, tokens)
    posterior_indices = posterior_logits.argmax(dim=-1)
    posterior_state = F.one_hot(
        posterior_indices, num_classes=world_model.n_classes
    ).float()
    h_z = world_model.join_h_and_z(h, posterior_state)
    next_z_embed = world_model.z_embedding(posterior_state.view(1, -1))
    return h, next_z_embed, h_z, prior_logits.argmax(dim=-1), posterior_indices


@torch.no_grad()
def _readouts(actor, critic, world_model, h_z, bins) -> dict[str, Any]:
    actor_probability = F.softmax(actor(h_z), dim=-1)
    return {
        "actor_probability": actor_probability,
        "actor_action": int(actor_probability.argmax(dim=-1).item()),
        "critic_value": float(twohot_expectation(critic(h_z), bins).item()),
        "reward_value": float(
            twohot_expectation(world_model.reward_predictor(h_z), bins).item()
        ),
        "continuation_probability": float(
            torch.sigmoid(world_model.continue_predictor(h_z)).item()
        ),
    }


def summarize_comparison(
    rows: list[dict], norm_inputs: dict[str, dict]
) -> tuple[dict, dict]:
    """Summarize downstream changes and apply the preregistered materiality gate."""
    metric_keys = (
        "feature_rms_difference",
        "actor_probability_l1",
        "critic_value_abs_difference",
        "reward_value_abs_difference",
        "continuation_probability_abs_difference",
    )
    downstream: dict[str, Any] = {
        key: distribution_summary([float(row[key]) for row in rows])
        for key in metric_keys
    }
    downstream.update(
        {
            "state_count": len(rows),
            "actor_action_disagreement_rate": float(
                np.mean([row["actor_action_disagreement"] for row in rows])
            ),
            "prior_category_disagreement_rate": float(
                np.mean([row["prior_category_disagreement"] for row in rows])
            ),
            "posterior_category_disagreement_rate": float(
                np.mean([row["posterior_category_disagreement"] for row in rows])
            ),
        }
    )

    local_p95 = [
        module["relative_output_scale_change"]["p95"]
        for module in norm_inputs.values()
        if module["relative_output_scale_change"]["p95"] is not None
    ]
    local_gate = bool(local_p95 and max(local_p95) >= 0.01)
    crossings = {
        "prior_category_disagreement_ge_0.01": (
            float(downstream["prior_category_disagreement_rate"]) >= 0.01
        ),
        "posterior_category_disagreement_ge_0.01": (
            float(downstream["posterior_category_disagreement_rate"]) >= 0.01
        ),
        "actor_action_disagreement_ge_0.01": (
            float(downstream["actor_action_disagreement_rate"]) >= 0.01
        ),
        "actor_probability_l1_p95_ge_0.02": (
            float(downstream["actor_probability_l1"]["p95"]) >= 0.02
        ),
        "critic_value_abs_difference_p95_ge_0.1": (
            float(downstream["critic_value_abs_difference"]["p95"]) >= 0.1
        ),
        "continuation_abs_difference_p95_ge_0.01": (
            float(downstream["continuation_probability_abs_difference"]["p95"])
            >= 0.01
        ),
    }
    downstream_gate = any(crossings.values())
    gate = {
        "local_norm_scale_p95_ge_0.01": local_gate,
        "downstream_threshold_crossings": crossings,
        "downstream_gate": downstream_gate,
        "materiality_gate_passed": local_gate and downstream_gate,
    }
    return downstream, gate


def run_epsilon_probe(
    source_checkpoint: Path,
    target_checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
) -> dict:
    """Compare checkpoint-faithful and reference-epsilon target computations."""
    (
        source_cfg,
        source_actor,
        _source_critic,
        _source_q,
        source_encoder,
        source_world_model,
        _source_data,
        _source_critic_key,
        _source_q_key,
    ) = load_checkpoint_models(source_checkpoint, device)
    current = load_checkpoint_models(target_checkpoint, device, critic_source="online")
    reference = load_checkpoint_models(target_checkpoint, device, critic_source="online")
    (
        target_cfg,
        current_actor,
        current_critic,
        _current_q,
        current_encoder,
        current_world_model,
        checkpoint_data,
        _current_critic_key,
        _current_q_key,
    ) = current
    (
        _reference_cfg,
        reference_actor,
        reference_critic,
        _reference_q,
        reference_encoder,
        reference_world_model,
        _reference_data,
        _reference_critic_key,
        _reference_q_key,
    ) = reference
    if (
        source_cfg.environment_name != target_cfg.environment_name
        or source_cfg.n_actions != target_cfg.n_actions
        or source_cfg.n_observations != target_cfg.n_observations
    ):
        raise ValueError("source and target checkpoints must use the same environment")

    recorder = RMSNormInputRecorder(current_world_model)
    changed_modules = set_reference_epsilon(reference_world_model)
    if not changed_modules or set(changed_modules) != set(recorder.mean_square):
        recorder.close()
        raise ValueError("current and reference copies expose different default RMSNorms")

    bins = symexp_twohot_bins(
        target_cfg.b_start,
        target_cfg.b_end,
        int(getattr(target_cfg, "num_bins", 255)),
        device=device,
    )
    env = gym.make(source_cfg.environment_name)
    spec = env.spec
    max_steps = int(spec.max_episode_steps or 500) if spec is not None else 500
    rows: list[dict] = []
    episode_returns: list[float] = []
    try:
        with torch.no_grad():
            for episode in range(episodes):
                obs, _ = env.reset(seed=seed + episode)
                source_h = torch.zeros(
                    1, source_cfg.d_hidden * source_cfg.rnn_n_blocks, device=device
                )
                current_h = torch.zeros(
                    1, target_cfg.d_hidden * target_cfg.rnn_n_blocks, device=device
                )
                reference_h = current_h.clone()
                source_previous_action = torch.zeros(
                    1, source_cfg.n_actions, device=device
                )
                target_previous_action = torch.zeros(
                    1, target_cfg.n_actions, device=device
                )
                source_z = torch.zeros(
                    1,
                    source_world_model.n_latents,
                    source_world_model.n_classes,
                    device=device,
                )
                target_z = torch.zeros(
                    1,
                    current_world_model.n_latents,
                    current_world_model.n_classes,
                    device=device,
                )
                source_z_embed = source_world_model.z_embedding(source_z.view(1, -1))
                current_z_embed = current_world_model.z_embedding(target_z.view(1, -1))
                reference_z_embed = reference_world_model.z_embedding(
                    target_z.view(1, -1)
                )
                episode_return = 0.0

                for timestep in range(max_steps):
                    state = np.asarray(obs, dtype=np.float32)
                    source_h, source_z_embed, source_h_z = _observe_state(
                        source_cfg,
                        source_encoder,
                        source_world_model,
                        source_h,
                        source_z_embed,
                        source_previous_action,
                        state,
                    )
                    (
                        current_h,
                        current_z_embed,
                        current_h_z,
                        current_prior,
                        current_posterior,
                    ) = _observe_target(
                        target_cfg,
                        current_encoder,
                        current_world_model,
                        current_h,
                        current_z_embed,
                        target_previous_action,
                        state,
                    )
                    (
                        reference_h,
                        reference_z_embed,
                        reference_h_z,
                        reference_prior,
                        reference_posterior,
                    ) = _observe_target(
                        target_cfg,
                        reference_encoder,
                        reference_world_model,
                        reference_h,
                        reference_z_embed,
                        target_previous_action,
                        state,
                    )
                    current_readouts = _readouts(
                        current_actor,
                        current_critic,
                        current_world_model,
                        current_h_z,
                        bins,
                    )
                    reference_readouts = _readouts(
                        reference_actor,
                        reference_critic,
                        reference_world_model,
                        reference_h_z,
                        bins,
                    )
                    rows.append(
                        {
                            "episode": episode,
                            "t": timestep,
                            "prior_category_disagreement": float(
                                (current_prior != reference_prior).float().mean().item()
                            ),
                            "posterior_category_disagreement": float(
                                (current_posterior != reference_posterior)
                                .float()
                                .mean()
                                .item()
                            ),
                            "feature_rms_difference": float(
                                (current_h_z - reference_h_z)
                                .square()
                                .mean()
                                .sqrt()
                                .item()
                            ),
                            "actor_action_current": current_readouts["actor_action"],
                            "actor_action_reference": reference_readouts[
                                "actor_action"
                            ],
                            "actor_action_disagreement": int(
                                current_readouts["actor_action"]
                                != reference_readouts["actor_action"]
                            ),
                            "actor_probability_l1": float(
                                (
                                    current_readouts["actor_probability"]
                                    - reference_readouts["actor_probability"]
                                )
                                .abs()
                                .sum()
                                .item()
                            ),
                            "critic_value_current": current_readouts["critic_value"],
                            "critic_value_reference": reference_readouts[
                                "critic_value"
                            ],
                            "critic_value_abs_difference": abs(
                                current_readouts["critic_value"]
                                - reference_readouts["critic_value"]
                            ),
                            "reward_value_abs_difference": abs(
                                current_readouts["reward_value"]
                                - reference_readouts["reward_value"]
                            ),
                            "continuation_probability_abs_difference": abs(
                                current_readouts["continuation_probability"]
                                - reference_readouts["continuation_probability"]
                            ),
                        }
                    )
                    source_action = int(
                        source_actor(source_h_z).argmax(dim=-1).item()
                    )
                    action = torch.tensor([source_action], device=device)
                    source_previous_action = F.one_hot(
                        action, num_classes=source_cfg.n_actions
                    ).float()
                    target_previous_action = F.one_hot(
                        action, num_classes=target_cfg.n_actions
                    ).float()
                    obs, reward, terminated, truncated, _ = env.step(source_action)
                    episode_return += float(reward)
                    if terminated or truncated:
                        break
                episode_returns.append(episode_return)
    finally:
        recorder.close()
        env.close()

    if not rows:
        raise ValueError("probe collected no states")
    norm_inputs = recorder.summary()
    downstream, gate = summarize_comparison(rows, norm_inputs)
    step = checkpoint_data.get("step", checkpoint_data.get("train_step"))
    summary = {
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "target_train_step": int(step) if step is not None else None,
        "device": device,
        "seed": seed,
        "episodes": episodes,
        "source_episode_return": distribution_summary(episode_returns),
        "local_epsilon": float(torch.finfo(torch.float32).eps),
        "reference_epsilon": REFERENCE_EPSILON,
        "changed_modules": changed_modules,
        "norm_inputs": norm_inputs,
        "downstream": downstream,
        "gate": gate,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/control_ablation/rmsnorm_epsilon")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = resolve_device(args.device)

    summaries = []
    for target in args.targets:
        target_name = f"{target.parent.parent.name}_{target.stem}"
        LOGGER.info("probing %s", target_name)
        summaries.append(
            run_epsilon_probe(
                args.source_checkpoint.resolve(),
                target.resolve(),
                args.out / target_name,
                device=device,
                episodes=args.episodes,
                seed=args.seed,
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"targets": summaries}, indent=2) + "\n"
    )
    LOGGER.info("wrote %s", args.out / "summary.json")


if __name__ == "__main__":
    main()
