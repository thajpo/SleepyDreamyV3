#!/usr/bin/env python3
"""Audit a CartPole actor on the states produced by its own policy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import learned_continue_discount, symlog, symexp
from dreamer.models.dreaming import enumerate_first_action_values
if __package__:
    from scripts.probe_cartpole_q import (
        action_preference,
        heuristic_action,
        load_checkpoint_models,
        one_step_outcome,
        pearson,
        rollout_score,
    )
else:
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        action_preference,
        heuristic_action,
        load_checkpoint_models,
        one_step_outcome,
        pearson,
        rollout_score,
    )


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def summarize_on_policy_rows(
    rows: list[dict],
    *,
    checkpoint_path: Path,
    train_step: int | None,
    critical_margin: float,
    terminal_window: int,
) -> dict:
    """Summarize trusted counterfactual decisions along completed episodes."""
    if not rows:
        raise ValueError("on-policy probe produced no states")

    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode"])].append(row)

    actionable = [row for row in rows if int(row["true_pref"]) >= 0]
    errors = [row for row in actionable if not bool(row["actor_correct"])]
    critical = [
        row for row in actionable if abs(float(row["true_delta"])) >= critical_margin
    ]
    critical_errors = [row for row in critical if not bool(row["actor_correct"])]
    terminal_actionable = [
        row for row in actionable if int(row["steps_to_end"]) <= terminal_window
    ]
    terminal_errors = [
        row for row in terminal_actionable if not bool(row["actor_correct"])
    ]
    avoidable_immediate_failures = [
        row
        for row in rows
        if bool(row["chosen_action_terminated"])
        and not bool(row["other_action_terminated"])
    ]
    q_rows = [row for row in actionable if "q_delta" in row]
    q_margined = [row for row in q_rows if abs(float(row["q_delta"])) > 1e-6]

    episode_returns = [float(ep_rows[0]["episode_return"]) for ep_rows in by_episode.values()]
    episode_error_counts = [
        sum(
            int(row["true_pref"]) >= 0 and not bool(row["actor_correct"])
            for row in ep_rows
        )
        for ep_rows in by_episode.values()
    ]
    episodes_with_terminal_error = sum(
        any(
            int(row["true_pref"]) >= 0
            and not bool(row["actor_correct"])
            and int(row["steps_to_end"]) <= terminal_window
            for row in ep_rows
        )
        for ep_rows in by_episode.values()
    )

    return {
        "checkpoint": str(checkpoint_path),
        "train_step": train_step,
        "episodes": len(by_episode),
        "states": len(rows),
        "mean_episode_return": float(np.mean(episode_returns)),
        "median_episode_return": float(np.median(episode_returns)),
        "min_episode_return": float(np.min(episode_returns)),
        "max_episode_return": float(np.max(episode_returns)),
        "solved_episode_fraction": _fraction(
            sum(value >= 500.0 for value in episode_returns), len(episode_returns)
        ),
        "actionable_states": len(actionable),
        "actor_vs_rollout_accuracy": _fraction(
            len(actionable) - len(errors), len(actionable)
        ),
        "q_vs_rollout_accuracy": _fraction(
            sum(int(row["q_pref"]) == int(row["true_pref"]) for row in q_rows),
            len(q_rows),
        ),
        "q_delta_true_delta_pearson": pearson(
            [float(row["q_delta"]) for row in q_rows],
            [float(row["true_delta"]) for row in q_rows],
        ),
        "actor_vs_q_accuracy": _fraction(
            sum(int(row["actor_action"]) == int(row["q_pref"]) for row in q_margined),
            len(q_margined),
        ),
        "critical_margin": critical_margin,
        "critical_states": len(critical),
        "critical_accuracy": _fraction(
            len(critical) - len(critical_errors), len(critical)
        ),
        "terminal_window": terminal_window,
        "terminal_window_actionable_states": len(terminal_actionable),
        "terminal_window_accuracy": _fraction(
            len(terminal_actionable) - len(terminal_errors),
            len(terminal_actionable),
        ),
        "avoidable_immediate_failures": len(avoidable_immediate_failures),
        "episodes_with_actionable_error": sum(count > 0 for count in episode_error_counts),
        "episodes_with_terminal_window_error": episodes_with_terminal_error,
        "mean_actionable_errors_per_episode": float(np.mean(episode_error_counts)),
        "mean_regret_per_episode": float(
            np.sum([float(row["trusted_regret"]) for row in rows]) / len(by_episode)
        ),
        "mean_wrong_actor_confidence": (
            float(np.mean([row["actor_confidence"] for row in errors]))
            if errors
            else None
        ),
        "mean_correct_actor_confidence": (
            float(
                np.mean(
                    [row["actor_confidence"] for row in actionable if row["actor_correct"]]
                )
            )
            if len(actionable) > len(errors)
            else None
        ),
        "action_hist": dict(Counter(str(row["actor_action"]) for row in rows)),
        "true_pref_hist": dict(
            Counter(str(row["true_pref"]) for row in actionable)
        ),
    }


def run_on_policy_probe(
    checkpoint_path: Path,
    out_dir: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
    rollout_horizon: int,
    model_horizon: int,
    critical_margin: float,
    terminal_window: int,
) -> dict:
    """Run the deployed deterministic actor and score each visited state."""
    cfg, actor, critic, _q_critic, encoder, world_model, checkpoint, _critic_key, _q_key = (
        load_checkpoint_models(checkpoint_path, device)
    )
    train_step = checkpoint.get("step", checkpoint.get("train_step"))

    env = gym.make(cfg.environment_name)
    score_env = gym.make(cfg.environment_name)
    env_spec = env.spec
    max_episode_steps = (
        int(env_spec.max_episode_steps)
        if env_spec is not None and env_spec.max_episode_steps is not None
        else 500
    )
    rows: list[dict] = []
    h_prev_backup = world_model.h_prev.clone()
    bins = symexp(
        torch.linspace(
            cfg.b_start,
            cfg.b_end,
            steps=int(getattr(cfg, "num_bins", 255)),
            device=device,
            dtype=torch.float32,
        )
    )
    imagination_discount = learned_continue_discount(
        cfg.gamma, bool(getattr(cfg, "contdisc", True))
    )

    try:
        with torch.no_grad():
            for episode in range(episodes):
                obs, _ = env.reset(seed=seed + episode)
                h = torch.zeros(
                    1, cfg.d_hidden * cfg.rnn_n_blocks, device=device
                )
                previous_action = torch.zeros(1, cfg.n_actions, device=device)
                z_prev = torch.zeros(
                    1, world_model.n_latents, world_model.n_classes, device=device
                )
                z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
                episode_rows: list[dict] = []
                episode_return = 0.0

                for timestep in range(max_episode_steps):
                    state = np.asarray(obs, dtype=np.float32)
                    h, _ = world_model.step_dynamics(
                        z_prev_embed, previous_action, h
                    )
                    tokens = encoder(
                        symlog(torch.from_numpy(state).to(device).float().unsqueeze(0))
                    )
                    posterior_logits = world_model.compute_posterior(h, tokens)
                    posterior_index = posterior_logits.argmax(dim=-1)
                    z_sample = F.one_hot(
                        posterior_index, num_classes=world_model.n_classes
                    ).float()
                    h_z = world_model.join_h_and_z(h, z_sample)
                    actor_logits = actor(h_z)
                    actor_probs = F.softmax(actor_logits, dim=-1)
                    action = int(actor_logits.argmax(dim=-1).item())
                    z_embed = world_model.z_embedding(z_sample.view(1, -1))
                    state_h_prev = world_model.h_prev.clone()
                    try:
                        q_values = enumerate_first_action_values(
                            h_z,
                            z_embed,
                            actor,
                            critic,
                            world_model,
                            cfg.n_actions,
                            cfg.d_hidden,
                            bins,
                            imagination_discount,
                            model_horizon,
                            terminal_reward_penalty=float(
                                getattr(cfg, "terminal_reward_penalty", 0.0)
                            ),
                        )
                    finally:
                        world_model.h_prev = state_h_prev

                    true_scores = [
                        rollout_score(score_env, state, candidate, rollout_horizon)
                        for candidate in range(cfg.n_actions)
                    ]
                    true_pref = action_preference(true_scores)
                    chosen_next, chosen_done = one_step_outcome(score_env, state, action)
                    del chosen_next
                    other_action = 1 - action
                    other_next, other_done = one_step_outcome(
                        score_env, state, other_action
                    )
                    del other_next
                    best_score = max(true_scores)
                    probs = actor_probs.squeeze(0).cpu().numpy()
                    q_values_np = q_values.squeeze(0).cpu().numpy()
                    row = {
                        "episode": episode,
                        "t": timestep,
                        "x": float(state[0]),
                        "x_dot": float(state[1]),
                        "theta": float(state[2]),
                        "theta_dot": float(state[3]),
                        "actor_action": action,
                        "actor_prob_0": float(probs[0]),
                        "actor_prob_1": float(probs[1]),
                        "actor_confidence": float(probs[action]),
                        "actor_entropy": float(
                            -np.sum(probs * np.log(probs + 1e-8))
                        ),
                        "q0": float(q_values_np[0]),
                        "q1": float(q_values_np[1]),
                        "q_delta": float(q_values_np[1] - q_values_np[0]),
                        "q_pref": int(q_values_np[1] > q_values_np[0]),
                        "true_score_0": float(true_scores[0]),
                        "true_score_1": float(true_scores[1]),
                        "true_delta": float(true_scores[1] - true_scores[0]),
                        "true_pref": int(true_pref),
                        "actor_correct": int(true_pref < 0 or action == true_pref),
                        "trusted_regret": float(best_score - true_scores[action]),
                        "heuristic_action": heuristic_action(state),
                        "chosen_action_terminated": int(chosen_done),
                        "other_action_terminated": int(other_done),
                    }

                    previous_action = F.one_hot(
                        torch.tensor([action], device=device),
                        num_classes=cfg.n_actions,
                    ).float()
                    z_prev_embed = world_model.z_embedding(z_sample.view(1, -1))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    episode_rows.append(row)
                    if terminated or truncated:
                        break

                episode_length = len(episode_rows)
                for row in episode_rows:
                    row["steps_to_end"] = episode_length - int(row["t"])
                    row["episode_return"] = episode_return
                rows.extend(episode_rows)
    finally:
        env.close()
        score_env.close()
        world_model.h_prev = h_prev_backup

    summary = summarize_on_policy_rows(
        rows,
        checkpoint_path=checkpoint_path,
        train_step=int(train_step) if train_step is not None else None,
        critical_margin=critical_margin,
        terminal_window=terminal_window,
    )
    summary.update(
        {
            "seed": seed,
            "rollout_horizon": rollout_horizon,
            "model_horizon": model_horizon,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/control_ablation/on_policy_probe")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--model-horizon", type=int, default=3)
    parser.add_argument("--critical-margin", type=float, default=15.0)
    parser.add_argument("--terminal-window", type=int, default=10)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.rollout_horizon <= 0:
        parser.error("--rollout-horizon must be positive")

    device = resolve_device(args.device)
    summaries = []
    for checkpoint in args.checkpoints:
        name = f"{checkpoint.parent.parent.name}_{checkpoint.stem}"
        print(f"Probing {name} on {device}...")
        summary = run_on_policy_probe(
            checkpoint.resolve(),
            args.out / name,
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            rollout_horizon=args.rollout_horizon,
            model_horizon=args.model_horizon,
            critical_margin=args.critical_margin,
            terminal_window=args.terminal_window,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
