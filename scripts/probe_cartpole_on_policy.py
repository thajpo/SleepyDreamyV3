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
from dreamer.models import (
    learned_continue_discount,
    symlog,
    symexp,
    symexp_twohot_bins,
)
from dreamer.models.dreaming import (
    enumerate_first_action_values,
    estimate_policy_lambda_action_values,
)
if __package__:
    from scripts.probe_cartpole_q import (
        action_preference,
        heuristic_action,
        hybrid_state_score,
        load_checkpoint_models,
        one_step_outcome,
        pearson,
        predict_one_step,
        rollout_score,
    )
else:
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        action_preference,
        heuristic_action,
        hybrid_state_score,
        load_checkpoint_models,
        one_step_outcome,
        pearson,
        predict_one_step,
        rollout_score,
    )


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _balanced_preference_accuracy(rows: list[dict], pred_key: str) -> float | None:
    """Average per-action recall so a constant preference cannot look accurate."""
    recalls = []
    for action in (0, 1):
        action_rows = [row for row in rows if int(row["true_pref"]) == action]
        if action_rows:
            recalls.append(
                sum(int(row[pred_key]) == action for row in action_rows)
                / len(action_rows)
            )
    return float(np.mean(recalls)) if recalls else None


def _sign_accuracy(predicted: list[float], actual: list[float]) -> float | None:
    pairs = [
        (pred, true)
        for pred, true in zip(predicted, actual)
        if abs(float(true)) > 1e-9
    ]
    return _fraction(
        sum((float(pred) > 0.0) == (float(true) > 0.0) for pred, true in pairs),
        len(pairs),
    )


def _posterior_reconstruct_next_state(
    world_model,
    encoder,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    action: int,
    next_state: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """Reconstruct a real next state after correcting the prior with its observation."""
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


def _preference_metrics(rows: list[dict], actionable: list[dict], prefix: str) -> dict:
    """Compare one predicted action-value component with trusted real rollouts."""
    delta_key = f"{prefix}_delta"
    pref_key = f"{prefix}_pref"
    component_rows = [
        row for row in actionable if delta_key in row and pref_key in row
    ]
    margined = [row for row in component_rows if abs(float(row[delta_key])) > 1e-6]
    all_component_rows = [row for row in rows if pref_key in row]
    return {
        f"{prefix}_margined_actionable_states": len(margined),
        f"{prefix}_vs_rollout_accuracy": _fraction(
            sum(int(row[pref_key]) == int(row["true_pref"]) for row in component_rows),
            len(component_rows),
        ),
        f"{prefix}_vs_rollout_balanced_accuracy": (
            _balanced_preference_accuracy(component_rows, pref_key)
        ),
        f"{prefix}_delta_true_delta_pearson": pearson(
            [float(row[delta_key]) for row in component_rows],
            [float(row["true_delta"]) for row in component_rows],
        ),
        f"actor_vs_{prefix}_accuracy": _fraction(
            sum(int(row["actor_action"]) == int(row[pref_key]) for row in margined),
            len(margined),
        ),
        f"mean_abs_{prefix}_delta": (
            float(np.mean([abs(float(row[delta_key])) for row in all_component_rows]))
            if all_component_rows
            else None
        ),
        f"{prefix}_pref_hist": dict(
            Counter(str(row[pref_key]) for row in all_component_rows)
        ),
    }


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
    episode_returns = [
        float(ep_rows[0]["episode_return"]) for ep_rows in by_episode.values()
    ]
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

    summary = {
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
        "actor_vs_rollout_balanced_accuracy": _balanced_preference_accuracy(
            actionable, "actor_action"
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
    prefixes = sorted(
        {
            key.removesuffix("_delta")
            for key in rows[0]
            if key.endswith("_delta")
            and (
                key in {"q_delta", "policy_q_delta"}
                or key.startswith("hybrid_")
                or key.startswith("decomp_")
            )
        }
    )
    for prefix in prefixes:
        summary.update(_preference_metrics(rows, actionable, prefix))
    policy_q_rows = [row for row in rows if "policy_q_delta_se" in row]
    if policy_q_rows:
        separated = [
            row
            for row in policy_q_rows
            if abs(float(row["policy_q_delta"]))
            > 1.96 * float(row["policy_q_delta_se"])
        ]
        separated_actionable = [
            row for row in separated if int(row["true_pref"]) >= 0
        ]
        summary.update(
            {
                "policy_q_confident_states": len(separated),
                "policy_q_confident_actor_agreement": _fraction(
                    sum(
                        int(row["actor_action"]) == int(row["policy_q_pref"])
                        for row in separated
                    ),
                    len(separated),
                ),
                "policy_q_confident_actionable_states": len(separated_actionable),
                "policy_q_confident_vs_rollout_balanced_accuracy": (
                    _balanced_preference_accuracy(
                        separated_actionable, "policy_q_pref"
                    )
                ),
            }
        )
    return summary


def run_on_policy_probe(
    checkpoint_path: Path,
    out_dir: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
    rollout_horizon: int,
    model_horizon: int,
    decomposition_horizons: list[int],
    critical_margin: float,
    terminal_window: int,
    policy_q_samples: int = 0,
    critic_source: str = "configured",
) -> dict:
    """Run the deployed deterministic actor and score each visited state."""
    cfg, actor, critic, _q_critic, encoder, world_model, checkpoint, critic_key, _q_key = (
        load_checkpoint_models(
            checkpoint_path,
            device,
            critic_source=critic_source,
        )
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
    bins = symexp_twohot_bins(
        cfg.b_start,
        cfg.b_end,
        int(getattr(cfg, "num_bins", 255)),
        device=device,
        dtype=torch.float32,
    )
    imagination_discount = learned_continue_discount(
        cfg.gamma, bool(getattr(cfg, "contdisc", True))
    )
    decomposition_horizons = sorted(
        {max(1, int(horizon)) for horizon in decomposition_horizons}
    )
    policy_q_generator = None
    if policy_q_samples:
        policy_q_generator = torch.Generator(device=device)
        policy_q_generator.manual_seed(seed + 2_000_000)

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
                        policy_q_values = None
                        policy_q_standard_errors = None
                        if policy_q_samples:
                            (
                                policy_q_values,
                                policy_q_standard_errors,
                            ) = estimate_policy_lambda_action_values(
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
                                policy_q_samples,
                                generator=policy_q_generator,
                                terminal_reward_penalty=float(
                                    getattr(cfg, "terminal_reward_penalty", 0.0)
                                ),
                            )
                        decomposition_values = {}
                        for horizon in decomposition_horizons:
                            full_values = (
                                q_values
                                if horizon == model_horizon
                                else enumerate_first_action_values(
                                    h_z,
                                    z_embed,
                                    actor,
                                    critic,
                                    world_model,
                                    cfg.n_actions,
                                    cfg.d_hidden,
                                    bins,
                                    imagination_discount,
                                    horizon,
                                    terminal_reward_penalty=float(
                                        getattr(cfg, "terminal_reward_penalty", 0.0)
                                    ),
                                )
                            )
                            model_return_values = enumerate_first_action_values(
                                h_z,
                                z_embed,
                                actor,
                                critic,
                                world_model,
                                cfg.n_actions,
                                cfg.d_hidden,
                                bins,
                                imagination_discount,
                                horizon,
                                terminal_reward_penalty=float(
                                    getattr(cfg, "terminal_reward_penalty", 0.0)
                                ),
                                bootstrap_value=False,
                            )
                            mode_full_values = enumerate_first_action_values(
                                h_z,
                                z_embed,
                                actor,
                                critic,
                                world_model,
                                cfg.n_actions,
                                cfg.d_hidden,
                                bins,
                                imagination_discount,
                                horizon,
                                terminal_reward_penalty=float(
                                    getattr(cfg, "terminal_reward_penalty", 0.0)
                                ),
                                latent_mode="mode",
                            )
                            mode_model_return_values = enumerate_first_action_values(
                                h_z,
                                z_embed,
                                actor,
                                critic,
                                world_model,
                                cfg.n_actions,
                                cfg.d_hidden,
                                bins,
                                imagination_discount,
                                horizon,
                                terminal_reward_penalty=float(
                                    getattr(cfg, "terminal_reward_penalty", 0.0)
                                ),
                                bootstrap_value=False,
                                latent_mode="mode",
                            )
                            decomposition_values[
                                f"decomp_full_q_h{horizon}"
                            ] = full_values
                            decomposition_values[
                                f"decomp_model_return_h{horizon}"
                            ] = model_return_values
                            decomposition_values[
                                f"decomp_mode_full_q_h{horizon}"
                            ] = mode_full_values
                            decomposition_values[
                                f"decomp_mode_model_return_h{horizon}"
                            ] = mode_model_return_values
                            if horizon == 1:
                                decomposition_values[
                                    "decomp_critic_bootstrap_h1"
                                ] = full_values - model_return_values
                                decomposition_values[
                                    "decomp_mode_critic_bootstrap_h1"
                                ] = mode_full_values - mode_model_return_values
                    finally:
                        world_model.h_prev = state_h_prev

                    true_scores = [
                        rollout_score(score_env, state, candidate, rollout_horizon)
                        for candidate in range(cfg.n_actions)
                    ]
                    true_pref = action_preference(true_scores)
                    real_next = []
                    real_done = []
                    predicted_next = []
                    predicted_continue = []
                    mode_predicted_next = []
                    mode_predicted_continue = []
                    posterior_reconstructed_next = []
                    state_h_prev = world_model.h_prev.clone()
                    try:
                        for candidate in range(cfg.n_actions):
                            next_state, done = one_step_outcome(
                                score_env, state, candidate
                            )
                            pred_state, continue_prob = predict_one_step(
                                world_model,
                                h,
                                z_embed,
                                candidate,
                                cfg.n_actions,
                            )
                            mode_pred_state, mode_continue_prob = predict_one_step(
                                world_model,
                                h,
                                z_embed,
                                candidate,
                                cfg.n_actions,
                                latent_mode="mode",
                            )
                            posterior_next_state = _posterior_reconstruct_next_state(
                                world_model,
                                encoder,
                                h,
                                z_embed,
                                candidate,
                                next_state,
                                cfg.n_actions,
                            )
                            real_next.append(next_state)
                            real_done.append(done)
                            predicted_next.append(pred_state)
                            predicted_continue.append(continue_prob)
                            mode_predicted_next.append(mode_pred_state)
                            mode_predicted_continue.append(mode_continue_prob)
                            posterior_reconstructed_next.append(
                                posterior_next_state
                            )
                    finally:
                        world_model.h_prev = state_h_prev
                    hybrid_state_scores = [
                        hybrid_state_score(
                            score_env, predicted_next[candidate], rollout_horizon
                        )
                        for candidate in range(cfg.n_actions)
                    ]
                    hybrid_continue_scores = [
                        1.0
                        + predicted_continue[candidate]
                        * (hybrid_state_scores[candidate] - 1.0)
                        for candidate in range(cfg.n_actions)
                    ]
                    hybrid_mode_state_scores = [
                        hybrid_state_score(
                            score_env, mode_predicted_next[candidate], rollout_horizon
                        )
                        for candidate in range(cfg.n_actions)
                    ]
                    hybrid_mode_continue_scores = [
                        1.0
                        + mode_predicted_continue[candidate]
                        * (hybrid_mode_state_scores[candidate] - 1.0)
                        for candidate in range(cfg.n_actions)
                    ]
                    chosen_done = real_done[action]
                    other_action = 1 - action
                    other_done = real_done[other_action]
                    best_score = max(true_scores)
                    probs = actor_probs.squeeze(0).cpu().numpy()
                    q_values_np = q_values.squeeze(0).cpu().numpy()
                    policy_q_values_np = (
                        None
                        if policy_q_values is None
                        else policy_q_values.squeeze(0).cpu().numpy()
                    )
                    policy_q_se_np = (
                        None
                        if policy_q_standard_errors is None
                        else policy_q_standard_errors.squeeze(0).cpu().numpy()
                    )
                    decomposition_np = {
                        key: values.squeeze(0).cpu().numpy()
                        for key, values in decomposition_values.items()
                    }
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
                        "hybrid_state_delta": float(
                            hybrid_state_scores[1] - hybrid_state_scores[0]
                        ),
                        "hybrid_state_pref": action_preference(hybrid_state_scores),
                        "hybrid_continue_delta": float(
                            hybrid_continue_scores[1] - hybrid_continue_scores[0]
                        ),
                        "hybrid_continue_pref": action_preference(
                            hybrid_continue_scores
                        ),
                        "hybrid_mode_state_delta": float(
                            hybrid_mode_state_scores[1]
                            - hybrid_mode_state_scores[0]
                        ),
                        "hybrid_mode_state_pref": action_preference(
                            hybrid_mode_state_scores
                        ),
                        "hybrid_mode_continue_delta": float(
                            hybrid_mode_continue_scores[1]
                            - hybrid_mode_continue_scores[0]
                        ),
                        "hybrid_mode_continue_pref": action_preference(
                            hybrid_mode_continue_scores
                        ),
                        "mean_one_step_state_mse": float(
                            0.5
                            * (
                                np.mean((predicted_next[0] - real_next[0]) ** 2)
                                + np.mean((predicted_next[1] - real_next[1]) ** 2)
                            )
                        ),
                        "mean_one_step_mode_state_mse": float(
                            0.5
                            * (
                                np.mean(
                                    (mode_predicted_next[0] - real_next[0]) ** 2
                                )
                                + np.mean(
                                    (mode_predicted_next[1] - real_next[1]) ** 2
                                )
                            )
                        ),
                        "mean_one_step_posterior_state_mse": float(
                            0.5
                            * (
                                np.mean(
                                    (posterior_reconstructed_next[0] - real_next[0])
                                    ** 2
                                )
                                + np.mean(
                                    (posterior_reconstructed_next[1] - real_next[1])
                                    ** 2
                                )
                            )
                        ),
                        "pred_continue_0": float(predicted_continue[0]),
                        "pred_continue_1": float(predicted_continue[1]),
                        "pred_delta_x_dot": float(
                            predicted_next[1][1] - predicted_next[0][1]
                        ),
                        "true_delta_x_dot": float(real_next[1][1] - real_next[0][1]),
                        "pred_delta_theta_dot": float(
                            predicted_next[1][3] - predicted_next[0][3]
                        ),
                        "true_delta_theta_dot": float(
                            real_next[1][3] - real_next[0][3]
                        ),
                        "mode_pred_delta_x_dot": float(
                            mode_predicted_next[1][1] - mode_predicted_next[0][1]
                        ),
                        "mode_pred_delta_theta_dot": float(
                            mode_predicted_next[1][3] - mode_predicted_next[0][3]
                        ),
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
                    if policy_q_values_np is not None and policy_q_se_np is not None:
                        policy_q_delta = float(
                            policy_q_values_np[1] - policy_q_values_np[0]
                        )
                        policy_q_delta_se = float(
                            np.sqrt(policy_q_se_np[0] ** 2 + policy_q_se_np[1] ** 2)
                        )
                        row.update(
                            {
                                "policy_q0": float(policy_q_values_np[0]),
                                "policy_q1": float(policy_q_values_np[1]),
                                "policy_q_delta": policy_q_delta,
                                "policy_q_delta_se": policy_q_delta_se,
                                "policy_q_pref": int(policy_q_delta > 0.0),
                            }
                        )
                    for state_index, state_name in enumerate(
                        ("x", "x_dot", "theta", "theta_dot")
                    ):
                        row[f"prior_{state_name}_mse"] = float(
                            0.5
                            * (
                                (
                                    predicted_next[0][state_index]
                                    - real_next[0][state_index]
                                )
                                ** 2
                                + (
                                    predicted_next[1][state_index]
                                    - real_next[1][state_index]
                                )
                                ** 2
                            )
                        )
                        row[f"mode_prior_{state_name}_mse"] = float(
                            0.5
                            * (
                                (
                                    mode_predicted_next[0][state_index]
                                    - real_next[0][state_index]
                                )
                                ** 2
                                + (
                                    mode_predicted_next[1][state_index]
                                    - real_next[1][state_index]
                                )
                                ** 2
                            )
                        )
                        row[f"posterior_{state_name}_mse"] = float(
                            0.5
                            * (
                                (
                                    posterior_reconstructed_next[0][state_index]
                                    - real_next[0][state_index]
                                )
                                ** 2
                                + (
                                    posterior_reconstructed_next[1][state_index]
                                    - real_next[1][state_index]
                                )
                                ** 2
                            )
                        )
                    for key, values in decomposition_np.items():
                        row.update(
                            {
                                f"{key}_delta": float(values[1] - values[0]),
                                f"{key}_pref": action_preference(values.tolist()),
                            }
                        )

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
            "decomposition_horizons": decomposition_horizons,
            "policy_q_samples": policy_q_samples,
            "policy_q_horizon": (
                int(cfg.num_dream_steps) if policy_q_samples else None
            ),
            "critic_key": critic_key,
            "mean_one_step_state_mse": float(
                np.mean([float(row["mean_one_step_state_mse"]) for row in rows])
            ),
            "mean_one_step_mode_state_mse": float(
                np.mean(
                    [float(row["mean_one_step_mode_state_mse"]) for row in rows]
                )
            ),
            "mean_one_step_posterior_state_mse": float(
                np.mean(
                    [
                        float(row["mean_one_step_posterior_state_mse"])
                        for row in rows
                    ]
                )
            ),
            "pred_delta_x_dot_true_pearson": pearson(
                [float(row["pred_delta_x_dot"]) for row in rows],
                [float(row["true_delta_x_dot"]) for row in rows],
            ),
            "pred_delta_theta_dot_true_pearson": pearson(
                [float(row["pred_delta_theta_dot"]) for row in rows],
                [float(row["true_delta_theta_dot"]) for row in rows],
            ),
            "pred_delta_x_dot_sign_accuracy": _sign_accuracy(
                [float(row["pred_delta_x_dot"]) for row in rows],
                [float(row["true_delta_x_dot"]) for row in rows],
            ),
            "pred_delta_theta_dot_sign_accuracy": _sign_accuracy(
                [float(row["pred_delta_theta_dot"]) for row in rows],
                [float(row["true_delta_theta_dot"]) for row in rows],
            ),
            "mode_pred_delta_x_dot_true_pearson": pearson(
                [float(row["mode_pred_delta_x_dot"]) for row in rows],
                [float(row["true_delta_x_dot"]) for row in rows],
            ),
            "mode_pred_delta_theta_dot_true_pearson": pearson(
                [float(row["mode_pred_delta_theta_dot"]) for row in rows],
                [float(row["true_delta_theta_dot"]) for row in rows],
            ),
            "mode_pred_delta_x_dot_sign_accuracy": _sign_accuracy(
                [float(row["mode_pred_delta_x_dot"]) for row in rows],
                [float(row["true_delta_x_dot"]) for row in rows],
            ),
            "mode_pred_delta_theta_dot_sign_accuracy": _sign_accuracy(
                [float(row["mode_pred_delta_theta_dot"]) for row in rows],
                [float(row["true_delta_theta_dot"]) for row in rows],
            ),
        }
    )
    for state_name in ("x", "x_dot", "theta", "theta_dot"):
        for source in ("prior", "mode_prior", "posterior"):
            key = f"{source}_{state_name}_mse"
            summary[f"mean_{key}"] = float(
                np.mean([float(row[key]) for row in rows])
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
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/control_ablation/on_policy_probe")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--model-horizon", type=int, default=3)
    parser.add_argument("--decompose-horizons", nargs="*", type=int, default=[1, 3])
    parser.add_argument("--critical-margin", type=float, default=15.0)
    parser.add_argument("--terminal-window", type=int, default=10)
    parser.add_argument("--policy-q-samples", type=int, default=0)
    parser.add_argument(
        "--critic-source",
        choices=("configured", "online", "slow"),
        default="configured",
        help="value head used for imagined action targets",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.rollout_horizon <= 0:
        parser.error("--rollout-horizon must be positive")
    if args.policy_q_samples == 1 or args.policy_q_samples < 0:
        parser.error("--policy-q-samples must be 0 or at least 2")

    device = resolve_device(args.device)
    summaries = []
    for checkpoint in args.checkpoints:
        name = f"{checkpoint.parent.parent.name}_{checkpoint.stem}"
        if args.critic_source != "configured":
            name = f"{name}_{args.critic_source}"
        print(f"Probing {name} on {device}...")
        summary = run_on_policy_probe(
            checkpoint.resolve(),
            args.out / name,
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            rollout_horizon=args.rollout_horizon,
            model_horizon=args.model_horizon,
            decomposition_horizons=args.decompose_horizons,
            critical_margin=args.critical_margin,
            terminal_window=args.terminal_window,
            policy_q_samples=args.policy_q_samples,
            critic_source=args.critic_source,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
