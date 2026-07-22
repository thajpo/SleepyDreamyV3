#!/usr/bin/env python3
"""Probe whether a CartPole Dreamer checkpoint's Q preferences match reality."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import infer_config_from_checkpoint, resolve_device
from dreamer.models import (
    initialize_actor,
    initialize_critic,
    initialize_q_critic,
    initialize_world_model,
    learned_continue_discount,
    symlog,
    symexp,
    symexp_twohot_bins,
    twohot_expectation,
    unimix_logits,
)
from dreamer.models.dreaming import enumerate_first_action_values


def heuristic_action(state: np.ndarray) -> int:
    """A compact stabilizing controller for CartPole's left/right action space."""
    x, x_dot, theta, theta_dot = [float(v) for v in state]
    score = 0.8 * x + 1.0 * x_dot + 6.0 * theta + 1.0 * theta_dot
    return int(score > 0.0)


def rollout_score(
    env: gym.Env,
    start_state: np.ndarray,
    first_action: int,
    horizon: int,
) -> float:
    """Return survival reward for one fixed first action, then heuristic control."""
    obs, _ = env.reset()
    del obs
    env.unwrapped.state = np.asarray(start_state, dtype=np.float64).copy()
    env.unwrapped.steps_beyond_terminated = None

    total = 0.0
    action = int(first_action)
    for step in range(max(1, int(horizon))):
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
        action = heuristic_action(np.asarray(obs, dtype=np.float32))
    return total


def heuristic_rollout_score(
    env: gym.Env,
    start_state: np.ndarray,
    horizon: int,
) -> float:
    """Return survival reward under heuristic control from an injected state."""
    if horizon <= 0:
        return 0.0

    obs, _ = env.reset()
    del obs
    state = np.asarray(start_state, dtype=np.float64).copy()
    env.unwrapped.state = state
    env.unwrapped.steps_beyond_terminated = None

    total = 0.0
    for _step in range(int(horizon)):
        action = heuristic_action(np.asarray(state, dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
        state = np.asarray(obs, dtype=np.float64)
    return total


def cartpole_state_is_terminal(env: gym.Env, state: np.ndarray) -> bool:
    """Apply CartPole's physical termination limits without taking another step."""
    x, _x_dot, theta, _theta_dot = [float(v) for v in state]
    base_env = env.unwrapped
    return bool(
        abs(x) > float(base_env.x_threshold)
        or abs(theta) > float(base_env.theta_threshold_radians)
    )


def hybrid_state_score(
    env: gym.Env,
    predicted_next_state: np.ndarray,
    horizon: int,
) -> float:
    """Score one learned transition using real dynamics for all later steps.

    CartPole awards one point for the modeled first transition. If the decoded
    next state is still physically valid, the trusted heuristic and real
    simulator score the remaining horizon. This isolates one-step transition
    quality from the learned critic and longer learned rollouts.
    """
    horizon = max(1, int(horizon))
    if cartpole_state_is_terminal(env, predicted_next_state):
        return 1.0
    return 1.0 + heuristic_rollout_score(
        env, predicted_next_state, horizon=horizon - 1
    )


def action_preference(scores: list[float]) -> int:
    """Return the preferred binary action, or -1 for an exact tie."""
    if math.isclose(scores[1], scores[0], rel_tol=0.0, abs_tol=1e-9):
        return -1
    return int(scores[1] > scores[0])


def one_step_outcome(
    env: gym.Env,
    start_state: np.ndarray,
    action: int,
) -> tuple[np.ndarray, bool]:
    """Return the real next state and done flag for one CartPole action."""
    obs, _ = env.reset()
    del obs
    env.unwrapped.state = np.asarray(start_state, dtype=np.float64).copy()
    env.unwrapped.steps_beyond_terminated = None
    next_obs, _reward, terminated, truncated, _ = env.step(int(action))
    return np.asarray(next_obs, dtype=np.float32), bool(terminated or truncated)


def predict_one_step(
    world_model,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    action: int,
    n_actions: int,
    latent_mode: str = "mean",
) -> tuple[np.ndarray, float]:
    """Decode the model's predicted next state after a fixed action."""
    if latent_mode not in {"mean", "mode"}:
        raise ValueError(f"unsupported latent_mode: {latent_mode}")
    action_onehot = F.one_hot(
        torch.tensor([int(action)], device=h.device), num_classes=n_actions
    ).float()
    h_next, prior_logits = world_model.step_dynamics(z_embed, action_onehot, h)
    prior_logits = unimix_logits(prior_logits, unimix_ratio=0.01)
    if latent_mode == "mode":
        z_state = F.one_hot(
            prior_logits.argmax(dim=-1), num_classes=world_model.n_classes
        ).float()
    else:
        z_state = F.softmax(prior_logits, dim=-1)
    h_z_next = world_model.join_h_and_z(h_next, z_state)
    pred_symlog_state = world_model.decoder(h_z_next)["state"]
    pred_state = symexp(pred_symlog_state).squeeze(0).detach().cpu().numpy()
    continue_prob = (
        torch.sigmoid(world_model.continue_predictor(h_z_next))
        .squeeze()
        .detach()
        .cpu()
        .item()
    )
    return pred_state.astype(np.float32), float(continue_prob)


def load_checkpoint_models(checkpoint_path: Path, device: str):
    cfg = infer_config_from_checkpoint(checkpoint_path, config_name=None)
    if cfg.environment_name != "CartPole-v1" or cfg.use_pixels:
        raise ValueError(f"{checkpoint_path} is not a state-only CartPole checkpoint")

    actor = initialize_actor(device, cfg)
    critic = initialize_critic(device, cfg)
    q_critic = initialize_q_critic(device, cfg)
    encoder, world_model = initialize_world_model(device, cfg, batch_size=1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    critic_key = (
        "critic_ema"
        if cfg.critic_slow_target and "critic_ema" in checkpoint
        else "critic"
    )
    critic.load_state_dict(checkpoint[critic_key])
    q_critic_key = None
    if "q_critic_ema" in checkpoint:
        q_critic.load_state_dict(checkpoint["q_critic_ema"])
        q_critic_key = "q_critic_ema"
    elif "q_critic" in checkpoint:
        q_critic.load_state_dict(checkpoint["q_critic"])
        q_critic_key = "q_critic"
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"], strict=False)

    actor.eval()
    critic.eval()
    q_critic.eval()
    encoder.eval()
    world_model.eval()
    return cfg, actor, critic, q_critic, encoder, world_model, checkpoint, critic_key, q_critic_key


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows: list[dict], checkpoint_path: Path, critic_key: str) -> dict:
    actionable = [r for r in rows if abs(r["true_delta"]) > 1e-9]
    q_margined = [r for r in actionable if abs(r["q_delta"]) > 1e-6]
    q_correct = [r for r in actionable if r["q_pref"] == r["true_pref"]]
    actor_correct = [r for r in actionable if r["actor_pref"] == r["true_pref"]]
    heuristic_correct = [r for r in actionable if r["heuristic_pref"] == r["true_pref"]]
    q_delta = [r["q_delta"] for r in actionable]
    true_delta = [r["true_delta"] for r in actionable]
    direct_q_rows = [r for r in actionable if "direct_q_delta" in r]
    direct_q_delta = [r["direct_q_delta"] for r in direct_q_rows]
    direct_q_true_delta = [r["true_delta"] for r in direct_q_rows]
    hybrid_state_correct = [
        r for r in actionable if r["hybrid_state_pref"] == r["true_pref"]
    ]
    hybrid_continue_correct = [
        r for r in actionable if r["hybrid_continue_pref"] == r["true_pref"]
    ]
    hybrid_state_margined = [
        r for r in actionable if abs(r["hybrid_state_delta"]) > 1e-6
    ]
    hybrid_continue_margined = [
        r for r in actionable if abs(r["hybrid_continue_delta"]) > 1e-6
    ]
    hybrid_state_delta = [r["hybrid_state_delta"] for r in actionable]
    hybrid_continue_delta = [r["hybrid_continue_delta"] for r in actionable]
    pred_delta_x_dot = [r["pred_delta_x_dot"] for r in rows]
    true_delta_x_dot = [r["true_delta_x_dot"] for r in rows]
    pred_delta_theta_dot = [r["pred_delta_theta_dot"] for r in rows]
    true_delta_theta_dot = [r["true_delta_theta_dot"] for r in rows]

    def sign_accuracy(pred: list[float], true: list[float]) -> float | None:
        pairs = [(p, t) for p, t in zip(pred, true) if abs(t) > 1e-9]
        if not pairs:
            return None
        return float(sum((p > 0.0) == (t > 0.0) for p, t in pairs) / len(pairs))

    def frac(count: int, denom: int) -> float | None:
        return None if denom == 0 else float(count / denom)

    summary = {
        "checkpoint": str(checkpoint_path),
        "critic_used": critic_key,
        "direct_q_critic_used": rows[0].get("direct_q_critic_used"),
        "states": len(rows),
        "actionable_states": len(actionable),
        "q_margined_actionable_states": len(q_margined),
        "hybrid_state_margined_actionable_states": len(hybrid_state_margined),
        "hybrid_continue_margined_actionable_states": len(
            hybrid_continue_margined
        ),
        "q_vs_rollout_accuracy": frac(len(q_correct), len(actionable)),
        "direct_q_vs_rollout_accuracy": frac(
            sum(r.get("direct_q_pref") == r["true_pref"] for r in direct_q_rows),
            len(direct_q_rows),
        ),
        "actor_vs_rollout_accuracy": frac(len(actor_correct), len(actionable)),
        "heuristic_vs_rollout_accuracy": frac(len(heuristic_correct), len(actionable)),
        "hybrid_state_vs_rollout_accuracy": frac(
            len(hybrid_state_correct), len(actionable)
        ),
        "hybrid_continue_vs_rollout_accuracy": frac(
            len(hybrid_continue_correct), len(actionable)
        ),
        "hybrid_state_vs_rollout_accuracy_margined": frac(
            sum(
                r["hybrid_state_pref"] == r["true_pref"]
                for r in hybrid_state_margined
            ),
            len(hybrid_state_margined),
        ),
        "hybrid_continue_vs_rollout_accuracy_margined": frac(
            sum(
                r["hybrid_continue_pref"] == r["true_pref"]
                for r in hybrid_continue_margined
            ),
            len(hybrid_continue_margined),
        ),
        "q_delta_true_delta_pearson": pearson(q_delta, true_delta),
        "direct_q_delta_true_delta_pearson": pearson(
            direct_q_delta, direct_q_true_delta
        ),
        "hybrid_state_delta_true_delta_pearson": pearson(
            hybrid_state_delta, true_delta
        ),
        "hybrid_continue_delta_true_delta_pearson": pearson(
            hybrid_continue_delta, true_delta
        ),
        "pred_delta_x_dot_true_pearson": pearson(
            pred_delta_x_dot, true_delta_x_dot
        ),
        "pred_delta_theta_dot_true_pearson": pearson(
            pred_delta_theta_dot, true_delta_theta_dot
        ),
        "pred_delta_x_dot_sign_accuracy": sign_accuracy(
            pred_delta_x_dot, true_delta_x_dot
        ),
        "pred_delta_theta_dot_sign_accuracy": sign_accuracy(
            pred_delta_theta_dot, true_delta_theta_dot
        ),
        "mean_one_step_state_mse": float(
            np.mean([0.5 * (r["next_mse_0"] + r["next_mse_1"]) for r in rows])
        ),
        "mean_one_step_state_mse_action0": float(
            np.mean([r["next_mse_0"] for r in rows])
        ),
        "mean_one_step_state_mse_action1": float(
            np.mean([r["next_mse_1"] for r in rows])
        ),
        "mean_pred_continue_done_action0": float(
            np.mean(
                [
                    r["pred_continue_0"]
                    for r in rows
                    if bool(r["real_done_0"])
                ]
                or [0.0]
            )
        ),
        "mean_pred_continue_done_action1": float(
            np.mean(
                [
                    r["pred_continue_1"]
                    for r in rows
                    if bool(r["real_done_1"])
                ]
                or [0.0]
            )
        ),
        "q_pref_hist": dict(Counter(str(r["q_pref"]) for r in rows)),
        "direct_q_pref_hist": dict(
            Counter(str(r["direct_q_pref"]) for r in rows if "direct_q_pref" in r)
        ),
        "actor_pref_hist": dict(Counter(str(r["actor_pref"]) for r in rows)),
        "hybrid_state_pref_hist": dict(
            Counter(str(r["hybrid_state_pref"]) for r in rows)
        ),
        "hybrid_continue_pref_hist": dict(
            Counter(str(r["hybrid_continue_pref"]) for r in rows)
        ),
        "true_pref_hist": dict(Counter(str(r["true_pref"]) for r in actionable)),
        "mean_abs_q_delta": float(np.mean([abs(r["q_delta"]) for r in rows])),
        "mean_abs_hybrid_state_delta": float(
            np.mean([abs(r["hybrid_state_delta"]) for r in rows])
        ),
        "mean_abs_hybrid_continue_delta": float(
            np.mean([abs(r["hybrid_continue_delta"]) for r in rows])
        ),
        "mean_abs_true_delta": float(np.mean([abs(r["true_delta"]) for r in rows])),
        "mean_actor_entropy": float(np.mean([r["actor_entropy"] for r in rows])),
        "mean_actor_prob_action1": float(np.mean([r["actor_prob_1"] for r in rows])),
    }
    decomposition_prefixes = sorted(
        key.removesuffix("_delta")
        for key in rows[0]
        if key.startswith("decomp_") and key.endswith("_delta")
    )
    for prefix in decomposition_prefixes:
        margined = [r for r in actionable if abs(r[f"{prefix}_delta"]) > 1e-6]
        correct = [r for r in actionable if r[f"{prefix}_pref"] == r["true_pref"]]
        prefix_delta = [r[f"{prefix}_delta"] for r in actionable]
        summary.update(
            {
                f"{prefix}_margined_actionable_states": len(margined),
                f"{prefix}_vs_rollout_accuracy": frac(len(correct), len(actionable)),
                f"{prefix}_vs_rollout_accuracy_margined": frac(
                    sum(r[f"{prefix}_pref"] == r["true_pref"] for r in margined),
                    len(margined),
                ),
                f"{prefix}_delta_true_delta_pearson": pearson(
                    prefix_delta, true_delta
                ),
                f"{prefix}_pref_hist": dict(
                    Counter(str(r[f"{prefix}_pref"]) for r in rows)
                ),
                f"mean_abs_{prefix}_delta": float(
                    np.mean([abs(r[f"{prefix}_delta"]) for r in rows])
                ),
            }
        )
    return summary


def run_probe(
    checkpoint_path: Path,
    out_dir: Path,
    device: str,
    states: int,
    seed: int,
    rollout_horizon: int,
    model_horizon: int,
    terminal_reward_penalty: float,
    decomposition_horizons: list[int] | None = None,
) -> dict:
    cfg, actor, critic, q_critic, encoder, world_model, checkpoint, critic_key, q_critic_key = (
        load_checkpoint_models(checkpoint_path, device)
    )
    del checkpoint

    rng = np.random.default_rng(seed)
    collect_env = gym.make(cfg.environment_name)
    score_env = gym.make(cfg.environment_name)
    obs, _ = collect_env.reset(seed=seed)

    h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
    prev_action_onehot = torch.zeros(1, cfg.n_actions, device=device)
    z_prev = torch.zeros(
        1, world_model.n_latents, world_model.n_classes, device=device
    )
    z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
    bins = symexp_twohot_bins(
        cfg.b_start,
        cfg.b_end,
        int(getattr(cfg, "num_bins", 255)),
        device=device,
        dtype=torch.float32,
    )
    decomposition_horizons = sorted(
        {max(1, int(horizon)) for horizon in (decomposition_horizons or [])}
    )
    imagination_discount = learned_continue_discount(
        cfg.gamma, bool(getattr(cfg, "contdisc", True))
    )

    rows: list[dict] = []
    episode = 0
    t = 0
    while len(rows) < states:
        state = np.asarray(obs, dtype=np.float32)
        state_vec = symlog(torch.from_numpy(state).to(device).float().unsqueeze(0))

        with torch.no_grad():
            h, prior_logits = world_model.step_dynamics(
                z_prev_embed, prev_action_onehot, h
            )
            del prior_logits
            tokens = encoder(state_vec)
            posterior_logits = world_model.compute_posterior(h, tokens)
            posterior_logits = unimix_logits(posterior_logits, unimix_ratio=0.01)
            posterior_idx = posterior_logits.argmax(dim=-1)
            z_sample = F.one_hot(
                posterior_idx, num_classes=world_model.n_classes
            ).float()
            h_z = world_model.join_h_and_z(h, z_sample)
            z_embed = world_model.z_embedding(z_sample.view(1, -1))
            actor_logits = unimix_logits(actor(h_z), unimix_ratio=0.01)
            actor_probs = F.softmax(actor_logits, dim=-1)
            actor_entropy = -torch.sum(
                actor_probs * torch.log(actor_probs + 1e-8), dim=-1
            )
            direct_q_np = None
            if q_critic_key is not None:
                direct_q_logits = q_critic(h_z).view(1, cfg.n_actions, -1)
                direct_q_values = twohot_expectation(direct_q_logits, bins)
                direct_q_np = direct_q_values.squeeze(0).detach().cpu().numpy()
            h_prev_backup = world_model.h_prev.clone()
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
                    terminal_reward_penalty=terminal_reward_penalty,
                )
                decomposition_values = {}
                for horizon in decomposition_horizons:
                    for label, objective, bootstrap_value in (
                        ("value_bootstrap", "value", True),
                        ("value_no_bootstrap", "value", False),
                        ("survival", "survival", False),
                    ):
                        key = f"decomp_{label}_h{horizon}"
                        decomposition_values[key] = enumerate_first_action_values(
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
                            terminal_reward_penalty=terminal_reward_penalty,
                            objective=objective,
                            bootstrap_value=bootstrap_value,
                        )
            finally:
                world_model.h_prev = h_prev_backup

        true_scores = [
            rollout_score(score_env, state, action, rollout_horizon)
            for action in range(cfg.n_actions)
        ]
        real_next = []
        real_done = []
        pred_next = []
        pred_continue = []
        h_prev_backup = world_model.h_prev.clone()
        try:
            for action in range(cfg.n_actions):
                next_state, done = one_step_outcome(score_env, state, action)
                pred_state, continue_prob = predict_one_step(
                    world_model, h, z_embed, action, cfg.n_actions
                )
                real_next.append(next_state)
                real_done.append(done)
                pred_next.append(pred_state)
                pred_continue.append(continue_prob)
        finally:
            world_model.h_prev = h_prev_backup

        hybrid_state_scores = [
            hybrid_state_score(score_env, pred_next[action], rollout_horizon)
            for action in range(cfg.n_actions)
        ]
        # Add the learned continuation head while retaining real downstream
        # dynamics. Subtracting one removes the shared modeled first-step reward.
        hybrid_continue_scores = [
            1.0
            + pred_continue[action] * (hybrid_state_scores[action] - 1.0)
            for action in range(cfg.n_actions)
        ]

        true_pref = action_preference(true_scores)
        hybrid_state_pref = action_preference(hybrid_state_scores)
        hybrid_continue_pref = action_preference(hybrid_continue_scores)

        q_np = q_values.squeeze(0).detach().cpu().numpy()
        decomposition_np = {
            key: values.squeeze(0).detach().cpu().numpy()
            for key, values in decomposition_values.items()
        }
        probs_np = actor_probs.squeeze(0).detach().cpu().numpy()
        row = {
                "episode": episode,
                "t": t,
                "x": float(state[0]),
                "x_dot": float(state[1]),
                "theta": float(state[2]),
                "theta_dot": float(state[3]),
                "q0": float(q_np[0]),
                "q1": float(q_np[1]),
                "q_delta": float(q_np[1] - q_np[0]),
                "q_pref": int(q_np[1] > q_np[0]),
                "actor_prob_0": float(probs_np[0]),
                "actor_prob_1": float(probs_np[1]),
                "actor_pref": int(probs_np[1] > probs_np[0]),
                "actor_entropy": float(actor_entropy.item()),
                "true_score_0": float(true_scores[0]),
                "true_score_1": float(true_scores[1]),
                "true_delta": float(true_scores[1] - true_scores[0]),
                "true_pref": int(true_pref),
                "heuristic_pref": int(heuristic_action(state)),
                "hybrid_state_score_0": float(hybrid_state_scores[0]),
                "hybrid_state_score_1": float(hybrid_state_scores[1]),
                "hybrid_state_delta": float(
                    hybrid_state_scores[1] - hybrid_state_scores[0]
                ),
                "hybrid_state_pref": int(hybrid_state_pref),
                "hybrid_continue_score_0": float(hybrid_continue_scores[0]),
                "hybrid_continue_score_1": float(hybrid_continue_scores[1]),
                "hybrid_continue_delta": float(
                    hybrid_continue_scores[1] - hybrid_continue_scores[0]
                ),
                "hybrid_continue_pref": int(hybrid_continue_pref),
                "next_mse_0": float(np.mean((pred_next[0] - real_next[0]) ** 2)),
                "next_mse_1": float(np.mean((pred_next[1] - real_next[1]) ** 2)),
                "real_done_0": int(real_done[0]),
                "real_done_1": int(real_done[1]),
                "pred_continue_0": float(pred_continue[0]),
                "pred_continue_1": float(pred_continue[1]),
                "decomp_continue_0": float(pred_continue[0]),
                "decomp_continue_1": float(pred_continue[1]),
                "decomp_continue_delta": float(
                    pred_continue[1] - pred_continue[0]
                ),
                "decomp_continue_pref": int(action_preference(pred_continue)),
                "pred_delta_x_dot": float(pred_next[1][1] - pred_next[0][1]),
                "true_delta_x_dot": float(real_next[1][1] - real_next[0][1]),
                "pred_delta_theta_dot": float(pred_next[1][3] - pred_next[0][3]),
                "true_delta_theta_dot": float(real_next[1][3] - real_next[0][3]),
            }
        for key, values in decomposition_np.items():
            row.update(
                {
                    f"{key}_0": float(values[0]),
                    f"{key}_1": float(values[1]),
                    f"{key}_delta": float(values[1] - values[0]),
                    f"{key}_pref": int(action_preference(values.tolist())),
                }
            )
        if direct_q_np is not None:
            row.update(
                {
                    "direct_q_critic_used": q_critic_key,
                    "direct_q0": float(direct_q_np[0]),
                    "direct_q1": float(direct_q_np[1]),
                    "direct_q_delta": float(direct_q_np[1] - direct_q_np[0]),
                    "direct_q_pref": int(direct_q_np[1] > direct_q_np[0]),
                }
            )
        rows.append(row)

        # Random collection keeps the probe independent of any one learned policy.
        action = int(rng.integers(0, cfg.n_actions))
        obs, _reward, terminated, truncated, _ = collect_env.step(action)
        prev_action_onehot = F.one_hot(
            torch.tensor([action], device=device), num_classes=cfg.n_actions
        ).float()
        z_prev_embed = z_embed.detach()
        t += 1

        if terminated or truncated:
            episode += 1
            t = 0
            obs, _ = collect_env.reset(seed=seed + episode)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            prev_action_onehot = torch.zeros(1, cfg.n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))

    collect_env.close()
    score_env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, checkpoint_path, critic_key)
    summary.update(
        {
            "seed": seed,
            "rollout_horizon": rollout_horizon,
            "model_horizon": model_horizon,
            "decomposition_horizons": decomposition_horizons,
            "terminal_reward_penalty": terminal_reward_penalty,
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/control_ablation/q_probe"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--model-horizon", type=int, default=3)
    parser.add_argument("--decompose-horizons", nargs="*", type=int, default=[])
    parser.add_argument("--terminal-reward-penalty", type=float, default=0.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    summaries = []
    for checkpoint in args.checkpoints:
        name = checkpoint.parent.parent.name
        print(f"Probing {name} on {device}...")
        summary = run_probe(
            checkpoint.resolve(),
            args.out / name,
            device=device,
            states=args.states,
            seed=args.seed,
            rollout_horizon=args.rollout_horizon,
            model_horizon=args.model_horizon,
            terminal_reward_penalty=args.terminal_reward_penalty,
            decomposition_horizons=args.decompose_horizons,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
