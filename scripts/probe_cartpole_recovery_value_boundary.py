#!/usr/bin/env python3
"""Locate the first broken value boundary on fixed CartPole recovery histories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import (
    learned_continue_discount,
    symexp_twohot_bins,
    twohot_expectation,
    unimix_logits,
)
from dreamer.models.dreaming import estimate_policy_lambda_action_values

if __package__:
    from scripts.probe_cartpole_checkpoint_drift import _observe_state, position_bin
    from scripts.probe_cartpole_q import (
        action_preference,
        load_checkpoint_models,
        pearson,
    )
else:
    from probe_cartpole_checkpoint_drift import (  # type: ignore[import-not-found]
        _observe_state,
        position_bin,
    )
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        action_preference,
        load_checkpoint_models,
        pearson,
    )


PREDICTION_KEYS = (
    "actor_action",
    "posterior_critic_pref",
    "prior_critic_pref",
    "policy_q_pref",
)


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _balanced_accuracy(rows: list[dict], prediction_key: str) -> float | None:
    recalls = []
    for action in (0, 1):
        action_rows = [row for row in rows if int(row["real_policy_pref"]) == action]
        if action_rows:
            recalls.append(
                sum(int(row[prediction_key]) == action for row in action_rows)
                / len(action_rows)
            )
    return float(np.mean(recalls)) if recalls else None


def summarize_boundary_rows(rows: list[dict]) -> dict:
    """Summarize action ordering at each model-to-behavior boundary."""
    if not rows:
        raise ValueError("recovery-value boundary probe produced no states")
    actionable = [row for row in rows if int(row["real_policy_pref"]) >= 0]
    summary: dict[str, object] = {
        "states": len(rows),
        "actionable_states": len(actionable),
        "real_policy_pref_hist": dict(
            Counter(str(row["real_policy_pref"]) for row in rows)
        ),
    }
    for key in PREDICTION_KEYS:
        prefix = key.removesuffix("_pref").removesuffix("_action")
        summary[f"{prefix}_vs_real_policy_accuracy"] = _fraction(
            sum(
                int(row[key]) == int(row["real_policy_pref"])
                for row in actionable
            ),
            len(actionable),
        )
        summary[f"{prefix}_vs_real_policy_balanced_accuracy"] = _balanced_accuracy(
            actionable, key
        )
        summary[f"{prefix}_pref_hist"] = dict(
            Counter(str(row[key]) for row in rows)
        )
        summary[f"{prefix}_actionable_pref_hist"] = dict(
            Counter(str(row[key]) for row in actionable)
        )

    preferred_probabilities = [
        float(row[f"actor_probability_{int(row['real_policy_pref'])}"])
        for row in actionable
        if f"actor_probability_{int(row['real_policy_pref'])}" in row
    ]
    if preferred_probabilities:
        summary["actor_preferred_action_probability_mean"] = float(
            np.mean(preferred_probabilities)
        )
        summary["actor_preferred_action_probability_median"] = float(
            np.median(preferred_probabilities)
        )
        summary["actor_preferred_action_probability_below_0_01"] = float(
            np.mean(np.asarray(preferred_probabilities) < 0.01)
        )

    for prefix in ("posterior_critic", "prior_critic", "policy_q"):
        summary[f"{prefix}_delta_real_policy_delta_pearson"] = pearson(
            [float(row[f"{prefix}_delta"]) for row in actionable],
            [float(row["real_policy_delta"]) for row in actionable],
        )
    summary["actor_vs_policy_q_accuracy"] = _fraction(
        sum(
            int(row["actor_action"]) == int(row["policy_q_pref"])
            for row in rows
            if abs(float(row["policy_q_delta"])) > 1e-9
        ),
        sum(abs(float(row["policy_q_delta"])) > 1e-9 for row in rows),
    )

    confident = [
        row
        for row in rows
        if abs(float(row["policy_q_delta"]))
        > 1.96 * float(row["policy_q_delta_se"])
    ]
    confident_actionable = [
        row for row in confident if int(row["real_policy_pref"]) >= 0
    ]
    summary.update(
        {
            "policy_q_confident_states": len(confident),
            "policy_q_confident_actionable_states": len(confident_actionable),
            "policy_q_confident_vs_real_policy_balanced_accuracy": (
                _balanced_accuracy(confident_actionable, "policy_q_pref")
            ),
            "actor_vs_policy_q_confident_accuracy": _fraction(
                sum(
                    int(row["actor_action"]) == int(row["policy_q_pref"])
                    for row in confident
                ),
                len(confident),
            ),
        }
    )
    return summary


def _sample_one_step_prior_values(
    h: torch.Tensor,
    z_embed: torch.Tensor,
    critic,
    world_model,
    bins: torch.Tensor,
    *,
    n_actions: int,
    imagination_discount: float,
    samples: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Estimate one-step learned reward/continue/critic values for each action."""
    device = h.device
    dtype = h.dtype
    action_ids = (
        torch.arange(n_actions, device=device)
        .view(n_actions, 1)
        .expand(n_actions, samples)
        .reshape(-1)
    )
    action_onehot = F.one_hot(action_ids, num_classes=n_actions).to(dtype=dtype)
    h_batch = h.expand(n_actions * samples, -1)
    z_batch = z_embed.expand(n_actions * samples, -1)
    h_next, prior_logits = world_model.step_dynamics(z_batch, action_onehot, h_batch)
    prior_probs = F.softmax(unimix_logits(prior_logits, unimix_ratio=0.01), dim=-1)
    flat_probs = prior_probs.reshape(-1, prior_probs.shape[-1])
    z_indices = torch.multinomial(flat_probs, 1, generator=generator).squeeze(-1)
    z_indices = z_indices.view(*prior_probs.shape[:-1])
    z_state = F.one_hot(z_indices, num_classes=world_model.n_classes).to(dtype=dtype)
    h_z = world_model.join_h_and_z(h_next, z_state)

    reward = twohot_expectation(world_model.reward_predictor(h_z), bins)
    continue_probability = torch.sigmoid(
        world_model.continue_predictor(h_z).squeeze(-1)
    )
    value = twohot_expectation(critic(h_z), bins)
    q_value = reward + imagination_discount * continue_probability * value

    def reshape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(n_actions, samples)

    q_samples = reshape(q_value)
    return {
        "q": q_samples.mean(dim=-1),
        "q_se": q_samples.std(dim=-1, unbiased=True) / samples**0.5,
        "reward": reshape(reward).mean(dim=-1),
        "continue": reshape(continue_probability).mean(dim=-1),
        "value": reshape(value).mean(dim=-1),
    }


def _inject_cartpole_state(env: gym.Env, state: np.ndarray) -> None:
    env.reset()
    base_env = cast(Any, env.unwrapped)
    base_env.state = np.asarray(state, dtype=np.float64).copy()
    base_env.steps_beyond_terminated = None


def _real_policy_branch(
    env: gym.Env,
    state: np.ndarray,
    first_action: int,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    cfg,
    actor,
    critic,
    encoder,
    world_model,
    bins: torch.Tensor,
    *,
    horizon: int,
) -> tuple[float, float]:
    """Force one action, then follow the target actor through real dynamics."""
    _inject_cartpole_state(env, state)
    local_h = h.clone()
    local_z_embed = z_embed.clone()
    action = int(first_action)
    score = 0.0
    posterior_q = 0.0

    for depth in range(horizon):
        next_obs, reward, terminated, truncated, _ = env.step(action)
        score += float(reward)
        done = bool(terminated or truncated)
        if depth == 0 and done:
            posterior_q = float(reward)
        if done:
            break

        previous_action = F.one_hot(
            torch.tensor([action], device=h.device), num_classes=cfg.n_actions
        ).float()
        local_h, local_z_embed, h_z = _observe_state(
            cfg,
            encoder,
            world_model,
            local_h,
            local_z_embed,
            previous_action,
            np.asarray(next_obs, dtype=np.float32),
        )
        if depth == 0:
            next_value = twohot_expectation(critic(h_z), bins)
            posterior_q = float(reward) + float(cfg.gamma) * float(next_value.item())
        action = int(actor(h_z).argmax(dim=-1).item())

    return score, posterior_q


@torch.no_grad()
def run_recovery_value_probe(
    source_checkpoint: Path,
    target_checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
    real_horizon: int,
    model_samples: int,
) -> dict:
    """Evaluate one target checkpoint on histories generated by one source."""
    (
        source_cfg,
        source_actor,
        _source_critic,
        _source_q_critic,
        source_encoder,
        source_world_model,
        _source_data,
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
        target_data,
        target_critic_key,
        _target_q_key,
    ) = load_checkpoint_models(target_checkpoint, device, critic_source="online")
    if (
        source_cfg.environment_name != target_cfg.environment_name
        or source_cfg.n_actions != target_cfg.n_actions
        or source_cfg.n_observations != target_cfg.n_observations
    ):
        raise ValueError("source and target checkpoints must use the same environment")

    bins = symexp_twohot_bins(
        target_cfg.b_start,
        target_cfg.b_end,
        int(target_cfg.num_bins),
        device=device,
        dtype=torch.float32,
    )
    imagination_discount = learned_continue_discount(
        target_cfg.gamma, bool(getattr(target_cfg, "contdisc", True))
    )
    prior_generator = torch.Generator(device=device).manual_seed(seed + 7_000_000)
    policy_generator = torch.Generator(device=device).manual_seed(seed + 8_000_000)
    history_env = gym.make(source_cfg.environment_name)
    branch_env = gym.make(source_cfg.environment_name)
    max_episode_steps = int(history_env.spec.max_episode_steps or 500)  # type: ignore[union-attr]
    rows: list[dict] = []
    source_h_backup = source_world_model.h_prev.clone()
    target_h_backup = target_world_model.h_prev.clone()

    try:
        for episode in range(episodes):
            obs, _ = history_env.reset(seed=seed + episode)
            source_h = torch.zeros(
                1, source_cfg.d_hidden * source_cfg.rnn_n_blocks, device=device
            )
            target_h = torch.zeros(
                1, target_cfg.d_hidden * target_cfg.rnn_n_blocks, device=device
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
                actor_logits = unimix_logits(
                    target_actor(target_h_z), unimix_ratio=0.01
                )
                actor_probabilities = F.softmax(actor_logits, dim=-1).squeeze(0)
                actor_action = int(actor_logits.argmax(dim=-1).item())

                h_prev_backup = target_world_model.h_prev.clone()
                try:
                    prior = _sample_one_step_prior_values(
                        target_h,
                        target_z_embed,
                        target_critic,
                        target_world_model,
                        bins,
                        n_actions=target_cfg.n_actions,
                        imagination_discount=imagination_discount,
                        samples=model_samples,
                        generator=prior_generator,
                    )
                    policy_q, policy_q_se = estimate_policy_lambda_action_values(
                        target_h_z,
                        target_z_embed,
                        target_actor,
                        target_critic,
                        target_world_model,
                        target_cfg.n_actions,
                        target_cfg.d_hidden,
                        bins,
                        imagination_discount,
                        target_cfg.lam,
                        target_cfg.num_dream_steps,
                        model_samples,
                        generator=policy_generator,
                        terminal_reward_penalty=float(
                            getattr(target_cfg, "terminal_reward_penalty", 0.0)
                        ),
                    )
                    real_scores = []
                    posterior_q = []
                    for candidate in range(target_cfg.n_actions):
                        score, q_value = _real_policy_branch(
                            branch_env,
                            state,
                            candidate,
                            target_h,
                            target_z_embed,
                            target_cfg,
                            target_actor,
                            target_critic,
                            target_encoder,
                            target_world_model,
                            bins,
                            horizon=real_horizon,
                        )
                        real_scores.append(score)
                        posterior_q.append(q_value)
                finally:
                    target_world_model.h_prev = h_prev_backup

                prior_q = prior["q"].cpu().numpy()
                prior_q_se = prior["q_se"].cpu().numpy()
                policy_values = policy_q.squeeze(0).cpu().numpy()
                policy_se = policy_q_se.squeeze(0).cpu().numpy()
                posterior_delta = posterior_q[1] - posterior_q[0]
                prior_delta = float(prior_q[1] - prior_q[0])
                policy_delta = float(policy_values[1] - policy_values[0])
                real_delta = real_scores[1] - real_scores[0]
                row = {
                    "episode": episode,
                    "t": timestep,
                    "x": float(state[0]),
                    "x_dot": float(state[1]),
                    "theta": float(state[2]),
                    "theta_dot": float(state[3]),
                    "abs_x_bin": position_bin(float(state[0])),
                    "source_action": source_action,
                    "actor_action": actor_action,
                    "actor_probability_0": float(actor_probabilities[0].item()),
                    "actor_probability_1": float(actor_probabilities[1].item()),
                    "real_policy_score_0": real_scores[0],
                    "real_policy_score_1": real_scores[1],
                    "real_policy_delta": real_delta,
                    "real_policy_pref": action_preference(real_scores),
                    "posterior_critic_q0": posterior_q[0],
                    "posterior_critic_q1": posterior_q[1],
                    "posterior_critic_delta": posterior_delta,
                    "posterior_critic_pref": int(posterior_delta > 0.0),
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
                for candidate in range(target_cfg.n_actions):
                    row[f"prior_reward_{candidate}"] = float(
                        prior["reward"][candidate].item()
                    )
                    row[f"prior_continue_{candidate}"] = float(
                        prior["continue"][candidate].item()
                    )
                    row[f"prior_value_{candidate}"] = float(
                        prior["value"][candidate].item()
                    )
                rows.append(row)

                action_tensor = torch.tensor([source_action], device=device)
                source_previous_action = F.one_hot(
                    action_tensor, num_classes=source_cfg.n_actions
                ).float()
                target_previous_action = F.one_hot(
                    action_tensor, num_classes=target_cfg.n_actions
                ).float()
                obs, _reward, terminated, truncated, _ = history_env.step(
                    source_action
                )
                if terminated or truncated:
                    break
    finally:
        history_env.close()
        branch_env.close()
        source_world_model.h_prev = source_h_backup
        target_world_model.h_prev = target_h_backup

    train_step = target_data.get("step", target_data.get("train_step"))
    summary = {
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "target_train_step": int(train_step) if train_step is not None else None,
        "critic_used": target_critic_key,
        "episodes": episodes,
        "seed_start": seed,
        "seed_end": seed + episodes - 1,
        "real_horizon": real_horizon,
        "model_samples": model_samples,
        **summarize_boundary_rows(rows),
    }
    summary["by_abs_x_bin"] = {}
    for label in ("0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"):
        bin_rows = [row for row in rows if row["abs_x_bin"] == label]
        if bin_rows:
            summary["by_abs_x_bin"][label] = summarize_boundary_rows(bin_rows)

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
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--real-horizon", type=int, default=30)
    parser.add_argument("--model-samples", type=int, default=64)
    args = parser.parse_args()
    if args.episodes <= 0 or args.real_horizon <= 0 or args.model_samples < 2:
        parser.error("episodes/horizon must be positive and model samples at least two")

    device = resolve_device(args.device)
    source_name = (
        f"{args.source_checkpoint.parent.parent.name}_{args.source_checkpoint.stem}"
    )
    for target in args.targets:
        target_name = f"{target.parent.parent.name}_{target.stem}"
        print(f"Fixed source {source_name}; probing target {target_name}...")
        summary = run_recovery_value_probe(
            args.source_checkpoint.resolve(),
            target.resolve(),
            args.out / f"source_{source_name}" / f"target_{target_name}",
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            real_horizon=args.real_horizon,
            model_samples=args.model_samples,
        )
        print(
            f"  step={summary['target_train_step']} "
            f"states={summary['states']} "
            f"actionable={summary['actionable_states']} "
            f"posterior={summary['posterior_critic_vs_real_policy_balanced_accuracy']} "
            f"prior={summary['prior_critic_vs_real_policy_balanced_accuracy']} "
            f"dream={summary['policy_q_vs_real_policy_balanced_accuracy']} "
            f"actor={summary['actor_vs_real_policy_balanced_accuracy']}"
        )


if __name__ == "__main__":
    main()
