#!/usr/bin/env python3
"""Decompose CartPole return error along matched real action sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import (
    learned_continue_discount,
    symexp,
    symexp_twohot_bins,
    twohot_expectation,
)
from dreamer.models.math_utils import unimix_logits

if __package__:
    from scripts.probe_cartpole_actor_supervision import observe_latent
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_actor_supervision import observe_latent  # type: ignore[import-not-found]
    from probe_cartpole_q import load_checkpoint_models  # type: ignore[import-not-found]


def discounted_return_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    """Return discounted finite-episode returns aligned with each reward."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + float(gamma) * running
        returns[index] = running
    return torch.tensor(returns, dtype=torch.float32)


def return_error_components(
    *,
    model_prefix: float,
    discounted_model_reward_prefix: float,
    predicted_discount: float,
    prior_bootstrap: float,
    actual_prefix: float,
    actual_discount: float,
    posterior_value: float,
    target_return: float,
) -> dict[str, float]:
    """Decompose model return error into oracle, prefix, and transport terms."""
    oracle_full = actual_prefix + actual_discount * posterior_value
    model_full = model_prefix + prior_bootstrap
    oracle_error = oracle_full - target_return
    reward_error = discounted_model_reward_prefix - actual_prefix
    continuation_prefix_error = model_prefix - discounted_model_reward_prefix
    final_discount_error = (
        predicted_discount - actual_discount
    ) * posterior_value
    critic_transport_error = (
        prior_bootstrap - predicted_discount * posterior_value
    )
    prefix_error = reward_error + continuation_prefix_error
    bootstrap_transport_error = (
        final_discount_error + critic_transport_error
    )
    rollout_error = model_full - oracle_full
    return {
        "target_return": target_return,
        "model_full": model_full,
        "oracle_full": oracle_full,
        "full_error": model_full - target_return,
        "oracle_error": oracle_error,
        "reward_error": reward_error,
        "continuation_prefix_error": continuation_prefix_error,
        "final_discount_error": final_discount_error,
        "critic_transport_error": critic_transport_error,
        "prefix_error": prefix_error,
        "bootstrap_transport_error": bootstrap_transport_error,
        "rollout_error": rollout_error,
    }


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "mae": float(np.abs(array).mean()),
        "rms": float(np.sqrt(np.mean(array**2))),
        "p90_abs": float(np.quantile(np.abs(array), 0.9)),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def optional_masked_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> float | None:
    """Return a masked mean, or None when the cohort is empty."""
    return float(values[mask].mean().item()) if bool(mask.any()) else None


@torch.no_grad()
def collect_real_episodes(
    cfg,
    actor,
    critic,
    encoder,
    world_model,
    bins: torch.Tensor,
    *,
    device: str,
    episodes: int,
    seed: int,
) -> list[dict]:
    """Collect deployed trajectories and their posterior critic values."""
    env = gym.make(cfg.environment_name)
    collected: list[dict] = []
    h_prev_backup = world_model.h_prev.clone()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            previous_action = torch.zeros(1, cfg.n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
            states: list[torch.Tensor] = []
            latents: list[torch.Tensor] = []
            z_embeds: list[torch.Tensor] = []
            actions: list[int] = []
            rewards: list[float] = []
            terminals: list[bool] = []
            truncated_episode = False

            while True:
                state = np.asarray(obs, dtype=np.float32)
                h_z, h, z_embed = observe_latent(
                    encoder,
                    world_model,
                    cfg,
                    state,
                    h,
                    previous_action,
                    z_prev_embed,
                )
                action = int(actor(h_z).argmax(dim=-1).item())
                next_obs, reward, terminated, truncated, _ = env.step(action)

                states.append(torch.from_numpy(state).to(device))
                latents.append(h_z.squeeze(0).detach())
                z_embeds.append(z_embed.squeeze(0).detach())
                actions.append(action)
                rewards.append(float(reward))
                terminals.append(bool(terminated))

                previous_action = F.one_hot(
                    torch.tensor([action], device=device), num_classes=cfg.n_actions
                ).float()
                z_prev_embed = z_embed.detach()
                obs = next_obs
                if terminated or truncated:
                    truncated_episode = bool(truncated)
                    break

            latent_tensor = torch.stack(latents)
            value_logits = critic(latent_tensor)
            posterior_values = twohot_expectation(value_logits, bins)
            collected.append(
                {
                    "episode": episode,
                    "states": torch.stack(states),
                    "latents": latent_tensor,
                    "z_embeds": torch.stack(z_embeds),
                    "actions": torch.tensor(actions, device=device),
                    "rewards": torch.tensor(rewards, device=device),
                    "terminals": torch.tensor(terminals, device=device),
                    "returns": discounted_return_to_go(rewards, cfg.gamma).to(device),
                    "posterior_values": posterior_values,
                    "truncated": truncated_episode,
                }
            )
    finally:
        env.close()
        world_model.h_prev = h_prev_backup
    return collected


@torch.no_grad()
def simulate_matched_prefix(
    initial_h_z: torch.Tensor,
    initial_z_embed: torch.Tensor,
    action_sequences: torch.Tensor,
    target_states: torch.Tensor,
    critic,
    world_model,
    bins: torch.Tensor,
    *,
    n_actions: int,
    d_hidden: int,
    gamma: float,
    imagination_discount: float,
    samples: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Replay fixed real actions through sampled prior dynamics in parallel."""
    batch_size, horizon = action_sequences.shape
    h_dim = world_model.n_blocks * d_hidden
    rollout_count = batch_size * samples
    h_prev_backup = world_model.h_prev.clone()

    def expand_samples(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.unsqueeze(1)
            .expand(batch_size, samples, *tensor.shape[1:])
            .reshape(rollout_count, *tensor.shape[1:])
        )

    def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
        flat = probs.reshape(-1, probs.shape[-1])
        draws = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        return draws.view(*probs.shape[:-1])

    h_z = expand_samples(initial_h_z)
    h_state = h_z[:, :h_dim]
    z_embed = expand_samples(initial_z_embed)
    prefix = torch.zeros(rollout_count, device=h_z.device)
    discounted_model_reward_prefix = torch.zeros(
        rollout_count, device=h_z.device
    )
    discount = torch.ones(rollout_count, device=h_z.device)

    for depth in range(horizon):
        action_ids = (
            action_sequences[:, depth]
            .unsqueeze(1)
            .expand(batch_size, samples)
            .reshape(rollout_count)
        )
        action_onehot = F.one_hot(action_ids, num_classes=n_actions).float()
        h_next, prior_logits = world_model.step_dynamics(
            z_embed, action_onehot, h_state
        )
        prior_probs = F.softmax(
            unimix_logits(prior_logits, unimix_ratio=0.01), dim=-1
        )
        z_indices = sample_categorical(prior_probs)
        z_state = F.one_hot(
            z_indices, num_classes=world_model.n_classes
        ).float()
        h_z = world_model.join_h_and_z(h_next, z_state)
        reward_logits = world_model.reward_predictor(h_z)
        reward = twohot_expectation(reward_logits, bins)
        continue_prob = torch.sigmoid(
            world_model.continue_predictor(h_z).squeeze(-1)
        )
        prefix = prefix + discount * reward
        discounted_model_reward_prefix = (
            discounted_model_reward_prefix
            + (gamma**depth) * reward
        )
        discount = discount * imagination_discount * continue_prob
        h_state = h_next
        z_embed = world_model.z_embedding(z_state.view(rollout_count, -1))

    prior_value_logits = critic(h_z)
    prior_values = twohot_expectation(prior_value_logits, bins)
    decoded_states = symexp(world_model.decoder(h_z)["state"])
    decoded_states = decoded_states.view(batch_size, samples, -1)
    target_states = target_states.unsqueeze(1)
    sample_state_mse = ((decoded_states - target_states) ** 2).mean(dim=-1)
    mean_state_mse = (
        (decoded_states.mean(dim=1) - target_states.squeeze(1)) ** 2
    ).mean(dim=-1)
    world_model.h_prev = h_prev_backup

    return {
        "model_prefix": prefix.view(batch_size, samples).mean(dim=-1),
        "discounted_model_reward_prefix": discounted_model_reward_prefix.view(
            batch_size, samples
        ).mean(dim=-1),
        "predicted_discount": discount.view(batch_size, samples).mean(dim=-1),
        "prior_value": prior_values.view(batch_size, samples).mean(dim=-1),
        "prior_bootstrap": (discount * prior_values)
        .view(batch_size, samples)
        .mean(dim=-1),
        "sample_state_mse": sample_state_mse.mean(dim=-1),
        "mean_state_mse": mean_state_mse,
    }


@torch.no_grad()
def continuation_calibration(
    episodes: list[dict],
    world_model,
    *,
    n_actions: int,
    d_hidden: int,
    samples: int,
    generator: torch.Generator,
    batch_size: int,
    gamma: float,
    imagination_discount: float,
) -> dict:
    """Measure one-step prior continuation on every real transition."""
    h_prev_backup = world_model.h_prev.clone()
    latent = torch.cat([episode["latents"] for episode in episodes])
    z_embed = torch.cat([episode["z_embeds"] for episode in episodes])
    actions = torch.cat([episode["actions"] for episode in episodes])
    terminals = torch.cat([episode["terminals"] for episode in episodes])
    predictions: list[torch.Tensor] = []
    h_dim = world_model.n_blocks * d_hidden

    for start in range(0, len(actions), batch_size):
        stop = min(len(actions), start + batch_size)
        count = stop - start
        h_state = (
            latent[start:stop, :h_dim]
            .unsqueeze(1)
            .expand(count, samples, h_dim)
            .reshape(count * samples, h_dim)
        )
        z_batch = (
            z_embed[start:stop]
            .unsqueeze(1)
            .expand(count, samples, z_embed.shape[-1])
            .reshape(count * samples, -1)
        )
        action_ids = (
            actions[start:stop]
            .unsqueeze(1)
            .expand(count, samples)
            .reshape(count * samples)
        )
        action_onehot = F.one_hot(action_ids, num_classes=n_actions).float()
        h_next, prior_logits = world_model.step_dynamics(
            z_batch, action_onehot, h_state
        )
        prior_probs = F.softmax(
            unimix_logits(prior_logits, unimix_ratio=0.01), dim=-1
        )
        flat = prior_probs.reshape(-1, prior_probs.shape[-1])
        z_indices = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        z_indices = z_indices.view(*prior_probs.shape[:-1])
        z_state = F.one_hot(
            z_indices, num_classes=world_model.n_classes
        ).float()
        h_z = world_model.join_h_and_z(h_next, z_state)
        predicted = torch.sigmoid(
            world_model.continue_predictor(h_z).squeeze(-1)
        ).view(count, samples).mean(dim=-1)
        predictions.append(predicted)

    predicted_continue = torch.cat(predictions)
    world_model.h_prev = h_prev_backup
    predicted_discount = imagination_discount * predicted_continue
    terminal_mask = terminals.bool()
    nonterminal_mask = ~terminal_mask
    target_discount = float(gamma) * nonterminal_mask.float()
    terminal_mean = optional_masked_mean(predicted_continue, terminal_mask)
    nonterminal_mean = optional_masked_mean(predicted_continue, nonterminal_mask)
    return {
        "transitions": int(len(predicted_continue)),
        "terminal_transitions": int(terminal_mask.sum().item()),
        "predicted_continue_terminal_mean": terminal_mean,
        "predicted_continue_nonterminal_mean": nonterminal_mean,
        "effective_discount_brier": float(
            ((predicted_discount - target_discount) ** 2).mean().item()
        ),
    }


def summarize_horizon_rows(rows: list[dict]) -> dict:
    """Summarize one fixed-horizon matched-rollout table."""
    summary: dict[str, object] = {
        key: _stats([float(row[key]) for row in rows])
        for key in (
            "full_error",
            "oracle_error",
            "rollout_error",
            "reward_error",
            "continuation_prefix_error",
            "final_discount_error",
            "critic_transport_error",
            "prefix_error",
            "bootstrap_transport_error",
        )
    }
    summary.update(
        {
            "count": len(rows),
            "model_target_pearson": _pearson(
                [float(row["model_full"]) for row in rows],
                [float(row["target_return"]) for row in rows],
            ),
            "mean_state_mse": float(
                np.mean([float(row["mean_state_mse"]) for row in rows])
            ),
            "sample_state_mse": float(
                np.mean([float(row["sample_state_mse"]) for row in rows])
            ),
            "predicted_discount_mean": float(
                np.mean([float(row["predicted_discount"]) for row in rows])
            ),
            "actual_discount": float(rows[0]["actual_discount"]),
        }
    )
    return summary


@torch.no_grad()
def run_probe(
    checkpoint_path: Path,
    out_dir: Path,
    *,
    device: str,
    episodes_count: int,
    seed: int,
    horizons: list[int],
    samples: int,
    batch_size: int,
) -> dict:
    cfg, actor, critic, _q_critic, encoder, world_model, checkpoint, critic_key, _q_key = (
        load_checkpoint_models(checkpoint_path, device)
    )
    train_step = checkpoint.get("step", checkpoint.get("train_step"))
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
    episodes = collect_real_episodes(
        cfg,
        actor,
        critic,
        encoder,
        world_model,
        bins,
        device=device,
        episodes=episodes_count,
        seed=seed,
    )
    posterior_errors = torch.cat(
        [episode["posterior_values"] - episode["returns"] for episode in episodes]
    )
    generator = torch.Generator(device=device).manual_seed(seed + 6_000_000)
    horizon_summaries = {}
    all_rows: dict[str, list[dict]] = {}

    for horizon in horizons:
        examples = [
            (episode, timestep)
            for episode in episodes
            for timestep in range(max(0, len(episode["actions"]) - horizon))
        ]
        rows: list[dict] = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            initial_h_z = torch.stack(
                [episode["latents"][timestep] for episode, timestep in chunk]
            )
            initial_z_embed = torch.stack(
                [episode["z_embeds"][timestep] for episode, timestep in chunk]
            )
            action_sequences = torch.stack(
                [
                    episode["actions"][timestep : timestep + horizon]
                    for episode, timestep in chunk
                ]
            )
            target_states = torch.stack(
                [episode["states"][timestep + horizon] for episode, timestep in chunk]
            )
            predicted = simulate_matched_prefix(
                initial_h_z,
                initial_z_embed,
                action_sequences,
                target_states,
                critic,
                world_model,
                bins,
                n_actions=cfg.n_actions,
                d_hidden=cfg.d_hidden,
                gamma=float(cfg.gamma),
                imagination_discount=imagination_discount,
                samples=samples,
                generator=generator,
            )
            for index, (episode, timestep) in enumerate(chunk):
                rewards = episode["rewards"][timestep : timestep + horizon]
                powers = torch.arange(horizon, device=device, dtype=torch.float32)
                gamma = torch.tensor(
                    float(cfg.gamma), device=device, dtype=torch.float32
                )
                actual_prefix = float(
                    torch.sum(rewards * torch.pow(gamma, powers)).item()
                )
                actual_discount = float(cfg.gamma) ** horizon
                predicted_discount = float(
                    predicted["predicted_discount"][index].item()
                )
                components = return_error_components(
                    model_prefix=float(predicted["model_prefix"][index].item()),
                    discounted_model_reward_prefix=float(
                        predicted["discounted_model_reward_prefix"][index].item()
                    ),
                    predicted_discount=predicted_discount,
                    prior_bootstrap=float(
                        predicted["prior_bootstrap"][index].item()
                    ),
                    actual_prefix=actual_prefix,
                    actual_discount=actual_discount,
                    posterior_value=float(
                        episode["posterior_values"][timestep + horizon].item()
                    ),
                    target_return=float(episode["returns"][timestep].item()),
                )
                rows.append(
                    {
                        "episode": int(episode["episode"]),
                        "t": timestep,
                        "horizon": horizon,
                        "actual_discount": actual_discount,
                        "predicted_discount": predicted_discount,
                        "mean_state_mse": float(
                            predicted["mean_state_mse"][index].item()
                        ),
                        "sample_state_mse": float(
                            predicted["sample_state_mse"][index].item()
                        ),
                        **components,
                    }
                )
        all_rows[str(horizon)] = rows
        horizon_summaries[str(horizon)] = summarize_horizon_rows(rows)

    continuation = continuation_calibration(
        episodes,
        world_model,
        n_actions=cfg.n_actions,
        d_hidden=cfg.d_hidden,
        samples=samples,
        generator=generator,
        batch_size=batch_size,
        gamma=float(cfg.gamma),
        imagination_discount=imagination_discount,
    )
    episode_returns = [float(episode["rewards"].sum().item()) for episode in episodes]
    summary = {
        "checkpoint": str(checkpoint_path),
        "train_step": train_step,
        "critic_source": critic_key,
        "device": device,
        "seed_start": seed,
        "episodes": episodes_count,
        "transitions": int(sum(len(episode["actions"]) for episode in episodes)),
        "truncated_episodes": int(sum(bool(episode["truncated"]) for episode in episodes)),
        "mean_episode_return": float(np.mean(episode_returns)),
        "horizons": horizons,
        "prior_samples": samples,
        "posterior_critic_error": _stats(posterior_errors.cpu().tolist()),
        "continuation": continuation,
        "horizon_metrics": horizon_summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "rows.json").write_text(json.dumps(all_rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10, 15])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.episodes <= 0 or args.samples < 2 or args.batch_size <= 0:
        parser.error("episodes and batch size must be positive; samples must be at least 2")
    horizons = sorted({int(horizon) for horizon in args.horizons})
    if not horizons or horizons[0] <= 0:
        parser.error("horizons must be positive")

    device = resolve_device(args.device)
    summaries = []
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.resolve()
        name = f"{checkpoint.parent.parent.name}_{checkpoint.stem}"
        print(f"Probing matched rollouts for {name} on {device}...")
        summary = run_probe(
            checkpoint,
            args.out / name,
            device=device,
            episodes_count=args.episodes,
            seed=args.seed,
            horizons=horizons,
            samples=args.samples,
            batch_size=args.batch_size,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
