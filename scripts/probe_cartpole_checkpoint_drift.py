#!/usr/bin/env python3
"""Compare CartPole checkpoints on histories fixed by one source policy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import (
    learned_continue_discount,
    symlog,
    symexp,
    symexp_twohot_bins,
)
from dreamer.models.dreaming import enumerate_first_action_values

if __package__:
    from scripts.probe_cartpole_q import (
        action_preference,
        load_checkpoint_models,
        one_step_outcome,
        pearson,
        predict_one_step,
        rollout_score,
    )
else:
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        action_preference,
        load_checkpoint_models,
        one_step_outcome,
        pearson,
        predict_one_step,
        rollout_score,
    )


STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")
POSITION_BINS = (
    (0.0, 0.5, "0.0-0.5"),
    (0.5, 1.0, "0.5-1.0"),
    (1.0, 1.5, "1.0-1.5"),
    (1.5, 2.0, "1.5-2.0"),
    (2.0, float("inf"), "2.0+"),
)


def position_bin(x: float) -> str:
    """Return a stable absolute-cart-position bucket label."""
    value = abs(float(x))
    for lower, upper, label in POSITION_BINS:
        if lower <= value < upper:
            return label
    raise AssertionError("position bins must cover every finite value")


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _balanced_accuracy(rows: list[dict], prediction_key: str) -> float | None:
    recalls = []
    for action in (0, 1):
        action_rows = [row for row in rows if int(row["true_pref"]) == action]
        if action_rows:
            recalls.append(
                sum(int(row[prediction_key]) == action for row in action_rows)
                / len(action_rows)
            )
    return float(np.mean(recalls)) if recalls else None


def _mean(rows: list[dict], key: str) -> float | None:
    return float(np.mean([float(row[key]) for row in rows])) if rows else None


def _diagnostic_metrics(rows: list[dict]) -> dict:
    actionable = [row for row in rows if int(row["true_pref"]) >= 0]
    q_margined = [row for row in actionable if abs(float(row["q_delta"])) > 1e-6]
    return {
        "states": len(rows),
        "actionable_states": len(actionable),
        "actor_vs_real_balanced_accuracy": _balanced_accuracy(
            actionable, "target_actor_action"
        ),
        "q_vs_real_balanced_accuracy": _balanced_accuracy(actionable, "q_pref"),
        "q_delta_true_delta_pearson": pearson(
            [float(row["q_delta"]) for row in actionable],
            [float(row["true_delta"]) for row in actionable],
        ),
        "actor_vs_q_accuracy": _fraction(
            sum(
                int(row["target_actor_action"]) == int(row["q_pref"])
                for row in q_margined
            ),
            len(q_margined),
        ),
        "mean_abs_q_delta": _mean(
            [{"value": abs(float(row["q_delta"]))} for row in rows], "value"
        ),
        "true_pref_hist": dict(Counter(str(row["true_pref"]) for row in rows)),
        "q_pref_hist": dict(Counter(str(row["q_pref"]) for row in rows)),
    }


def summarize_fixed_history_rows(
    rows: list[dict],
    *,
    source_checkpoint: Path,
    target_checkpoint: Path,
    train_step: int | None,
) -> dict:
    """Summarize model/value quality on one checkpoint-independent history set."""
    if not rows:
        raise ValueError("fixed-history probe produced no states")

    episode_returns = {}
    for row in rows:
        episode_returns[int(row["episode"])] = float(row["source_episode_return"])
    abs_x = np.asarray([abs(float(row["x"])) for row in rows], dtype=np.float64)

    summary = {
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "target_train_step": train_step,
        "episodes": len(episode_returns),
        "mean_source_episode_return": float(np.mean(list(episode_returns.values()))),
        "mean_abs_x": float(np.mean(abs_x)),
        "p90_abs_x": float(np.quantile(abs_x, 0.90)),
        "p99_abs_x": float(np.quantile(abs_x, 0.99)),
        "max_abs_x": float(np.max(abs_x)),
        "fraction_abs_x_ge_1": float(np.mean(abs_x >= 1.0)),
        "fraction_abs_x_ge_2": float(np.mean(abs_x >= 2.0)),
    }
    summary.update(_diagnostic_metrics(rows))

    for state_name in STATE_NAMES:
        for prefix in ("current_posterior", "one_step_prior", "one_step_posterior"):
            key = f"{prefix}_{state_name}_mse"
            summary[f"mean_{key}"] = _mean(rows, key)

    summary["by_abs_x_bin"] = {}
    for _lower, _upper, label in POSITION_BINS:
        bin_rows = [row for row in rows if row["abs_x_bin"] == label]
        if not bin_rows:
            continue
        metrics = _diagnostic_metrics(bin_rows)
        metrics.update(
            {
                "mean_current_posterior_x_mse": _mean(
                    bin_rows, "current_posterior_x_mse"
                ),
                "mean_one_step_prior_x_mse": _mean(
                    bin_rows, "one_step_prior_x_mse"
                ),
                "mean_one_step_posterior_x_mse": _mean(
                    bin_rows, "one_step_posterior_x_mse"
                ),
            }
        )
        summary["by_abs_x_bin"][label] = metrics
    return summary


def _observe_state(cfg, encoder, world_model, h, z_embed, previous_action, state):
    h, _prior_logits = world_model.step_dynamics(z_embed, previous_action, h)
    tokens = encoder(
        symlog(torch.from_numpy(state).to(h.device).float().unsqueeze(0))
    )
    posterior_logits = world_model.compute_posterior(h, tokens)
    posterior_state = F.one_hot(
        posterior_logits.argmax(dim=-1), num_classes=world_model.n_classes
    ).float()
    h_z = world_model.join_h_and_z(h, posterior_state)
    next_z_embed = world_model.z_embedding(posterior_state.view(1, -1))
    return h, next_z_embed, h_z


def _posterior_reconstruct_next_state(
    world_model,
    encoder,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    action: int,
    next_state: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    action_onehot = F.one_hot(
        torch.tensor([int(action)], device=h.device), num_classes=n_actions
    ).float()
    h_next, _prior_logits = world_model.step_dynamics(z_embed, action_onehot, h)
    tokens = encoder(
        symlog(torch.from_numpy(next_state).to(h.device).float().unsqueeze(0))
    )
    posterior_logits = world_model.compute_posterior(h_next, tokens)
    posterior_state = F.one_hot(
        posterior_logits.argmax(dim=-1), num_classes=world_model.n_classes
    ).float()
    h_z_next = world_model.join_h_and_z(h_next, posterior_state)
    reconstructed = symexp(world_model.decoder(h_z_next)["state"])
    return reconstructed.squeeze(0).cpu().numpy().astype(np.float32)


def run_fixed_history_probe(
    source_checkpoint: Path,
    target_checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
    rollout_horizon: int,
    model_horizon: int,
) -> dict:
    """Drive CartPole with one actor while evaluating a target checkpoint."""
    (
        source_cfg,
        source_actor,
        _source_critic,
        _source_q_critic,
        source_encoder,
        source_world_model,
        _source_checkpoint_data,
        _source_critic_key,
        _source_q_key,
    ) = load_checkpoint_models(source_checkpoint, device)
    (
        target_cfg,
        target_actor,
        target_critic,
        _target_q_critic,
        target_encoder,
        target_world_model,
        target_checkpoint_data,
        _target_critic_key,
        _target_q_key,
    ) = load_checkpoint_models(target_checkpoint, device)
    if (
        source_cfg.environment_name != target_cfg.environment_name
        or source_cfg.n_actions != target_cfg.n_actions
        or source_cfg.n_observations != target_cfg.n_observations
    ):
        raise ValueError("source and target checkpoints must use the same environment")

    env = gym.make(source_cfg.environment_name)
    score_env = gym.make(source_cfg.environment_name)
    env_spec = env.spec
    max_episode_steps = (
        int(env_spec.max_episode_steps)
        if env_spec is not None and env_spec.max_episode_steps is not None
        else 500
    )
    bins = symexp_twohot_bins(
        target_cfg.b_start,
        target_cfg.b_end,
        int(getattr(target_cfg, "num_bins", 255)),
        device=device,
        dtype=torch.float32,
    )
    imagination_discount = learned_continue_discount(
        target_cfg.gamma, bool(getattr(target_cfg, "contdisc", True))
    )
    rows: list[dict] = []
    source_h_backup = source_world_model.h_prev.clone()
    target_h_backup = target_world_model.h_prev.clone()

    try:
        with torch.no_grad():
            for episode in range(episodes):
                obs, _ = env.reset(seed=seed + episode)
                source_h = torch.zeros(
                    1,
                    source_cfg.d_hidden * source_cfg.rnn_n_blocks,
                    device=device,
                )
                target_h = torch.zeros(
                    1,
                    target_cfg.d_hidden * target_cfg.rnn_n_blocks,
                    device=device,
                )
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
                    target_world_model.n_latents,
                    target_world_model.n_classes,
                    device=device,
                )
                source_z_embed = source_world_model.z_embedding(source_z.view(1, -1))
                target_z_embed = target_world_model.z_embedding(target_z.view(1, -1))
                episode_rows: list[dict] = []
                episode_return = 0.0

                for timestep in range(max_episode_steps):
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
                    target_h, target_z_embed, target_h_z = _observe_state(
                        target_cfg,
                        target_encoder,
                        target_world_model,
                        target_h,
                        target_z_embed,
                        target_previous_action,
                        state,
                    )
                    source_action = int(
                        source_actor(source_h_z).argmax(dim=-1).item()
                    )
                    target_actor_action = int(
                        target_actor(target_h_z).argmax(dim=-1).item()
                    )
                    target_h_prev = target_world_model.h_prev.clone()
                    try:
                        q_values = enumerate_first_action_values(
                            target_h_z,
                            target_z_embed,
                            target_actor,
                            target_critic,
                            target_world_model,
                            target_cfg.n_actions,
                            target_cfg.d_hidden,
                            bins,
                            imagination_discount,
                            model_horizon,
                            terminal_reward_penalty=float(
                                getattr(target_cfg, "terminal_reward_penalty", 0.0)
                            ),
                        )
                        current_reconstruction = symexp(
                            target_world_model.decoder(target_h_z)["state"]
                        ).squeeze(0).cpu().numpy().astype(np.float32)
                        real_next = []
                        predicted_next = []
                        posterior_next = []
                        for candidate in range(target_cfg.n_actions):
                            next_state, _done = one_step_outcome(
                                score_env, state, candidate
                            )
                            predicted_state, _continue_probability = predict_one_step(
                                target_world_model,
                                target_h,
                                target_z_embed,
                                candidate,
                                target_cfg.n_actions,
                            )
                            reconstructed_state = _posterior_reconstruct_next_state(
                                target_world_model,
                                target_encoder,
                                target_h,
                                target_z_embed,
                                candidate,
                                next_state,
                                target_cfg.n_actions,
                            )
                            real_next.append(next_state)
                            predicted_next.append(predicted_state)
                            posterior_next.append(reconstructed_state)
                    finally:
                        target_world_model.h_prev = target_h_prev

                    true_scores = [
                        rollout_score(score_env, state, candidate, rollout_horizon)
                        for candidate in range(target_cfg.n_actions)
                    ]
                    true_pref = action_preference(true_scores)
                    q_values_np = q_values.squeeze(0).cpu().numpy()
                    row = {
                        "episode": episode,
                        "t": timestep,
                        "x": float(state[0]),
                        "x_dot": float(state[1]),
                        "theta": float(state[2]),
                        "theta_dot": float(state[3]),
                        "abs_x_bin": position_bin(float(state[0])),
                        "source_action": source_action,
                        "target_actor_action": target_actor_action,
                        "q0": float(q_values_np[0]),
                        "q1": float(q_values_np[1]),
                        "q_delta": float(q_values_np[1] - q_values_np[0]),
                        "q_pref": int(q_values_np[1] > q_values_np[0]),
                        "true_score_0": float(true_scores[0]),
                        "true_score_1": float(true_scores[1]),
                        "true_delta": float(true_scores[1] - true_scores[0]),
                        "true_pref": int(true_pref),
                    }
                    for state_index, state_name in enumerate(STATE_NAMES):
                        row[f"current_posterior_{state_name}_mse"] = float(
                            (current_reconstruction[state_index] - state[state_index])
                            ** 2
                        )
                        row[f"one_step_prior_{state_name}_mse"] = float(
                            np.mean(
                                [
                                    (
                                        predicted_next[candidate][state_index]
                                        - real_next[candidate][state_index]
                                    )
                                    ** 2
                                    for candidate in range(target_cfg.n_actions)
                                ]
                            )
                        )
                        row[f"one_step_posterior_{state_name}_mse"] = float(
                            np.mean(
                                [
                                    (
                                        posterior_next[candidate][state_index]
                                        - real_next[candidate][state_index]
                                    )
                                    ** 2
                                    for candidate in range(target_cfg.n_actions)
                                ]
                            )
                        )

                    action_tensor = torch.tensor([source_action], device=device)
                    source_previous_action = F.one_hot(
                        action_tensor, num_classes=source_cfg.n_actions
                    ).float()
                    target_previous_action = F.one_hot(
                        action_tensor, num_classes=target_cfg.n_actions
                    ).float()
                    obs, reward, terminated, truncated, _ = env.step(source_action)
                    episode_return += float(reward)
                    episode_rows.append(row)
                    if terminated or truncated:
                        break

                for row in episode_rows:
                    row["source_episode_return"] = episode_return
                rows.extend(episode_rows)
    finally:
        env.close()
        score_env.close()
        source_world_model.h_prev = source_h_backup
        target_world_model.h_prev = target_h_backup

    train_step = target_checkpoint_data.get(
        "step", target_checkpoint_data.get("train_step")
    )
    summary = summarize_fixed_history_rows(
        rows,
        source_checkpoint=source_checkpoint,
        target_checkpoint=target_checkpoint,
        train_step=int(train_step) if train_step is not None else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/control_ablation/checkpoint_drift")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--model-horizon", type=int, default=3)
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    summaries = []
    source_name = (
        f"{args.source_checkpoint.parent.parent.name}_"
        f"{args.source_checkpoint.stem}"
    )
    for target in args.targets:
        target_name = f"{target.parent.parent.name}_{target.stem}"
        out_dir = args.out / f"source_{source_name}" / f"target_{target_name}"
        print(f"Fixed source {source_name}; probing target {target_name}...")
        summary = run_fixed_history_probe(
            args.source_checkpoint.resolve(),
            target.resolve(),
            out_dir,
            device=args.device,
            episodes=args.episodes,
            seed=args.seed,
            rollout_horizon=args.rollout_horizon,
            model_horizon=args.model_horizon,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    source_summary_path = args.out / f"source_{source_name}_summary.json"
    source_summary_path.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
