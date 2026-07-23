#!/usr/bin/env python3
"""Test whether collapsed CartPole posterior features can fit real recovery values."""

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
    initialize_critic,
    symlog,
    symexp_twohot_bins,
    twohot_expectation,
)

if __package__:
    from scripts.probe_cartpole_checkpoint_drift import _observe_state
    from scripts.probe_cartpole_critic_supervision import (
        episode_split,
        initialize_state_critic,
        train_critic,
    )
    from scripts.probe_cartpole_q import (
        action_preference,
        heuristic_action,
        load_checkpoint_models,
        pearson,
    )
else:
    from probe_cartpole_checkpoint_drift import _observe_state  # type: ignore[import-not-found]
    from probe_cartpole_critic_supervision import (  # type: ignore[import-not-found]
        episode_split,
        initialize_state_critic,
        train_critic,
    )
    from probe_cartpole_q import (  # type: ignore[import-not-found]
        action_preference,
        heuristic_action,
        load_checkpoint_models,
        pearson,
    )


def _inject_state(env: gym.Env, state: np.ndarray) -> None:
    env.reset()
    base_env = cast(Any, env.unwrapped)
    base_env.state = np.asarray(state, dtype=np.float64).copy()
    base_env.steps_beyond_terminated = None


def q_from_continuation(reward: float, gamma: float, continuation: float) -> float:
    """Align a successor-state value with the action that reached it."""
    return float(reward) + float(gamma) * float(continuation)


@torch.no_grad()
def _target_policy_continuation(
    env: gym.Env,
    start_state: np.ndarray,
    h: torch.Tensor,
    z_embed: torch.Tensor,
    h_z: torch.Tensor,
    cfg,
    actor,
    encoder,
    world_model,
    *,
    horizon: int,
) -> float:
    """Discounted real return from a successor under the target actor."""
    if horizon <= 0:
        return 0.0
    _inject_state(env, start_state)
    total = 0.0
    discount = 1.0
    local_h = h.clone()
    local_z_embed = z_embed.clone()
    local_h_z = h_z.clone()
    for _ in range(horizon):
        action = int(actor(local_h_z).argmax(dim=-1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += discount * float(reward)
        if terminated or truncated:
            break
        action_onehot = F.one_hot(
            torch.tensor([action], device=h.device), num_classes=cfg.n_actions
        ).float()
        local_h, local_z_embed, local_h_z = _observe_state(
            cfg,
            encoder,
            world_model,
            local_h,
            local_z_embed,
            action_onehot,
            np.asarray(obs, dtype=np.float32),
        )
        discount *= float(cfg.gamma)
    return total


def _heuristic_continuation(
    env: gym.Env,
    start_state: np.ndarray,
    gamma: float,
    *,
    horizon: int,
) -> float:
    """Discounted real return from a successor under fixed heuristic control."""
    if horizon <= 0:
        return 0.0
    _inject_state(env, start_state)
    state = np.asarray(start_state, dtype=np.float32)
    total = 0.0
    discount = 1.0
    for _ in range(horizon):
        action = heuristic_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += discount * float(reward)
        if terminated or truncated:
            break
        state = np.asarray(obs, dtype=np.float32)
        discount *= float(gamma)
    return total


def _balanced_accuracy(real: list[int], predicted: list[int]) -> float | None:
    recalls = []
    for action in (0, 1):
        indices = [index for index, value in enumerate(real) if value == action]
        if indices:
            recalls.append(
                sum(predicted[index] == action for index in indices) / len(indices)
            )
    return float(np.mean(recalls)) if recalls else None


def paired_action_metrics(
    rows: list[dict],
    q_key: str,
    *,
    policy: str,
) -> dict:
    """Summarize paired binary action ordering for one continuation policy."""
    by_base: dict[int, dict[int, dict]] = {}
    for row in rows:
        by_base.setdefault(int(row["base_index"]), {})[int(row["action"])] = row

    real_preferences: list[int] = []
    predicted_preferences: list[int] = []
    real_margins: list[float] = []
    predicted_margins: list[float] = []
    for candidates in by_base.values():
        if set(candidates) != {0, 1}:
            continue
        real_q = [float(candidates[action][f"{policy}_q"]) for action in (0, 1)]
        predicted_q = [float(candidates[action][q_key]) for action in (0, 1)]
        real_pref = action_preference(real_q)
        predicted_pref = action_preference(predicted_q)
        if real_pref < 0:
            continue
        real_preferences.append(real_pref)
        predicted_preferences.append(predicted_pref)
        real_margins.append(real_q[1] - real_q[0])
        predicted_margins.append(predicted_q[1] - predicted_q[0])

    return {
        "actionable_pairs": len(real_preferences),
        "real_preference_hist": dict(Counter(str(x) for x in real_preferences)),
        "predicted_preference_hist": dict(
            Counter(str(x) for x in predicted_preferences)
        ),
        "balanced_accuracy": _balanced_accuracy(
            real_preferences, predicted_preferences
        ),
        "margin_pearson": pearson(predicted_margins, real_margins),
        "mean_absolute_margin_error": (
            float(
                np.mean(
                    np.abs(
                        np.asarray(predicted_margins) - np.asarray(real_margins)
                    )
                )
            )
            if real_margins
            else None
        ),
    }


@torch.no_grad()
def collect_recovery_dataset(
    source_checkpoint: Path,
    target_checkpoint: Path,
    *,
    device: str,
    minimum_states: int,
    seed: int,
    horizon: int,
):
    """Collect complete source episodes and paired real successor targets."""
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
    (
        target_cfg,
        target_actor,
        target_critic,
        _target_q,
        target_encoder,
        target_world_model,
        target_data,
        target_critic_key,
        _target_q_key,
    ) = load_checkpoint_models(target_checkpoint, device, critic_source="online")
    if (
        source_cfg.environment_name != target_cfg.environment_name
        or source_cfg.n_actions != target_cfg.n_actions
    ):
        raise ValueError("source and target checkpoints must use the same environment")

    history_env = gym.make(source_cfg.environment_name)
    step_env = gym.make(source_cfg.environment_name)
    target_env = gym.make(source_cfg.environment_name)
    heuristic_env = gym.make(source_cfg.environment_name)
    max_episode_steps = int(history_env.spec.max_episode_steps or 500)  # type: ignore[union-attr]
    bins = symexp_twohot_bins(
        target_cfg.b_start,
        target_cfg.b_end,
        target_cfg.num_bins,
        device=device,
    )
    rows: list[dict] = []
    latent_features: list[torch.Tensor] = []
    successor_states: list[torch.Tensor] = []
    episode = 0
    base_index = 0
    source_h_backup = source_world_model.h_prev.clone()
    target_h_backup = target_world_model.h_prev.clone()

    try:
        while base_index < minimum_states:
            obs, _ = history_env.reset(seed=seed + episode)
            source_h = torch.zeros(
                1, source_cfg.d_hidden * source_cfg.rnn_n_blocks, device=device
            )
            target_h = torch.zeros(
                1, target_cfg.d_hidden * target_cfg.rnn_n_blocks, device=device
            )
            source_action = torch.zeros(1, source_cfg.n_actions, device=device)
            target_action = torch.zeros(1, target_cfg.n_actions, device=device)
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
                    source_action,
                    state,
                )
                target_h, target_z_embed, _target_h_z = _observe_state(
                    target_cfg,
                    target_encoder,
                    target_world_model,
                    target_h,
                    target_z_embed,
                    target_action,
                    state,
                )

                h_backup = target_world_model.h_prev.clone()
                for candidate in range(target_cfg.n_actions):
                    _inject_state(step_env, state)
                    next_obs, reward, terminated, truncated, _ = step_env.step(candidate)
                    next_state = np.asarray(next_obs, dtype=np.float32)
                    done = bool(terminated or truncated)
                    candidate_action = F.one_hot(
                        torch.tensor([candidate], device=device),
                        num_classes=target_cfg.n_actions,
                    ).float()
                    next_h, next_z_embed, next_h_z = _observe_state(
                        target_cfg,
                        target_encoder,
                        target_world_model,
                        target_h,
                        target_z_embed,
                        candidate_action,
                        next_state,
                    )
                    if done:
                        target_continuation = 0.0
                        heuristic_continuation = 0.0
                        deployed_q = float(reward)
                    else:
                        target_continuation = _target_policy_continuation(
                            target_env,
                            next_state,
                            next_h,
                            next_z_embed,
                            next_h_z,
                            target_cfg,
                            target_actor,
                            target_encoder,
                            target_world_model,
                            horizon=horizon - 1,
                        )
                        heuristic_continuation = _heuristic_continuation(
                            heuristic_env,
                            next_state,
                            target_cfg.gamma,
                            horizon=horizon - 1,
                        )
                        deployed_value = twohot_expectation(
                            target_critic(next_h_z),
                            bins,
                        )
                        deployed_q = float(reward) + float(target_cfg.gamma) * float(
                            deployed_value.item()
                        )

                    row_index = len(rows)
                    rows.append(
                        {
                            "row_index": row_index,
                            "base_index": base_index,
                            "episode": episode,
                            "t": timestep,
                            "action": candidate,
                            "done": done,
                            "reward": float(reward),
                            "target_continuation_value": target_continuation,
                            "heuristic_continuation_value": heuristic_continuation,
                            "target_q": q_from_continuation(
                                float(reward), target_cfg.gamma, target_continuation
                            ),
                            "heuristic_q": q_from_continuation(
                                float(reward),
                                target_cfg.gamma,
                                heuristic_continuation,
                            ),
                            "deployed_q": deployed_q,
                            "next_x": float(next_state[0]),
                            "next_x_dot": float(next_state[1]),
                            "next_theta": float(next_state[2]),
                            "next_theta_dot": float(next_state[3]),
                        }
                    )
                    latent_features.append(next_h_z.squeeze(0).detach().cpu())
                    successor_states.append(torch.from_numpy(next_state.copy()))
                    target_world_model.h_prev = h_backup

                chosen_action = int(source_actor(source_h_z).argmax(dim=-1).item())
                action_tensor = torch.tensor([chosen_action], device=device)
                source_action = F.one_hot(
                    action_tensor, num_classes=source_cfg.n_actions
                ).float()
                target_action = F.one_hot(
                    action_tensor, num_classes=target_cfg.n_actions
                ).float()
                obs, _reward, terminated, truncated, _ = history_env.step(chosen_action)
                base_index += 1
                if terminated or truncated:
                    break
            episode += 1
    finally:
        for env in (history_env, step_env, target_env, heuristic_env):
            env.close()
        source_world_model.h_prev = source_h_backup
        target_world_model.h_prev = target_h_backup

    return (
        target_cfg,
        target_data,
        target_critic_key,
        rows,
        torch.stack(latent_features).to(device),
        symlog(torch.stack(successor_states).to(device)),
    )


def run_probe(
    source_checkpoint: Path,
    target_checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    minimum_states: int,
    seed: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    """Collect one checkpoint dataset and fit paired frozen-feature critics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    (
        cfg,
        checkpoint,
        critic_key,
        rows,
        latent_features,
        true_states,
    ) = collect_recovery_dataset(
        source_checkpoint,
        target_checkpoint,
        device=device,
        minimum_states=minimum_states,
        seed=seed,
        horizon=horizon,
    )
    episode_ids = torch.tensor([int(row["episode"]) for row in rows], device=device)
    train_indices, test_indices = episode_split(episode_ids, 0.2, seed)
    bins = symexp_twohot_bins(cfg.b_start, cfg.b_end, cfg.num_bins, device=device)
    held_out = set(test_indices.cpu().tolist())
    test_rows = [row for index, row in enumerate(rows) if index in held_out]
    results: dict[str, dict] = {}

    for policy in ("target", "heuristic"):
        targets = torch.tensor(
            [float(row[f"{policy}_continuation_value"]) for row in rows],
            device=device,
        )
        live = torch.tensor([not bool(row["done"]) for row in rows], device=device)
        fit_train = train_indices[live[train_indices]]
        fit_test = test_indices[live[test_indices]]

        torch.manual_seed(seed)
        latent_critic = initialize_critic(device, cfg)
        latent_fit = train_critic(
            latent_critic,
            latent_features,
            targets,
            bins,
            fit_train,
            fit_test,
            epochs,
            batch_size,
            lr,
        )
        torch.manual_seed(seed)
        state_critic = initialize_state_critic(cfg, device)
        state_fit = train_critic(
            state_critic,
            true_states,
            targets,
            bins,
            fit_train,
            fit_test,
            epochs,
            batch_size,
            lr,
        )

        with torch.no_grad():
            latent_values = twohot_expectation(latent_critic(latent_features), bins)
            state_values = twohot_expectation(state_critic(true_states), bins)
        for index, row in enumerate(rows):
            if bool(row["done"]):
                row[f"fresh_latent_{policy}_q"] = float(row["reward"])
                row[f"fresh_state_{policy}_q"] = float(row["reward"])
            else:
                row[f"fresh_latent_{policy}_q"] = q_from_continuation(
                    row["reward"], cfg.gamma, latent_values[index].item()
                )
                row[f"fresh_state_{policy}_q"] = q_from_continuation(
                    row["reward"], cfg.gamma, state_values[index].item()
                )

        results[policy] = {
            "fresh_latent_value_fit": latent_fit,
            "fresh_state_value_fit": state_fit,
            "fresh_latent_action_ordering": paired_action_metrics(
                test_rows, f"fresh_latent_{policy}_q", policy=policy
            ),
            "fresh_state_action_ordering": paired_action_metrics(
                test_rows, f"fresh_state_{policy}_q", policy=policy
            ),
            "deployed_action_ordering": (
                paired_action_metrics(test_rows, "deployed_q", policy=policy)
                if policy == "target"
                else None
            ),
        }

    train_step = checkpoint.get("step", checkpoint.get("train_step"))
    summary = {
        "source_checkpoint": str(source_checkpoint.resolve()),
        "target_checkpoint": str(target_checkpoint.resolve()),
        "target_train_step": int(train_step) if train_step is not None else None,
        "critic_used": critic_key,
        "base_states": len(rows) // cfg.n_actions,
        "rows": len(rows),
        "episodes": len(set(int(row["episode"]) for row in rows)),
        "seed_start": seed,
        "horizon": horizon,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "results": results,
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
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.states <= 0 or args.horizon <= 1 or args.epochs <= 0:
        parser.error("states/epochs must be positive and horizon must exceed one")

    device = resolve_device(args.device)
    source_name = (
        f"{args.source_checkpoint.parent.parent.name}_{args.source_checkpoint.stem}"
    )
    for target in args.targets:
        target_name = f"{target.parent.parent.name}_{target.stem}"
        print(f"Fixed source {source_name}; probing target {target_name}...")
        summary = run_probe(
            args.source_checkpoint,
            target,
            args.out / f"source_{source_name}" / f"target_{target_name}",
            device=device,
            minimum_states=args.states,
            seed=args.seed,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        target_result = summary["results"]["target"]
        print(
            f"  step={summary['target_train_step']} bases={summary['base_states']} "
            f"latent_corr={target_result['fresh_latent_value_fit']['test']['pearson']} "
            f"state_corr={target_result['fresh_state_value_fit']['test']['pearson']} "
            f"latent_bal={target_result['fresh_latent_action_ordering']['balanced_accuracy']} "
            f"deployed_bal={target_result['deployed_action_ordering']['balanced_accuracy']}"
        )


if __name__ == "__main__":
    main()
