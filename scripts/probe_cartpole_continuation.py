#!/usr/bin/env python3
"""Compare CartPole continuation predictions under posterior and prior latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import learned_continue_discount, symlog
from dreamer.models.math_utils import unimix_logits

if __package__:
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_q import load_checkpoint_models  # type: ignore[import-not-found]


def binary_roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Return tie-aware ROC AUC for scores where larger means more positive."""
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def continuation_error_components(
    prior_discount: float,
    posterior_discount: float,
    target_discount: float,
) -> dict[str, float]:
    """Decompose prior error through the observation-conditioned posterior."""
    posterior_error = posterior_discount - target_discount
    transport_error = prior_discount - posterior_discount
    return {
        "prior_error": prior_discount - target_discount,
        "posterior_error": posterior_error,
        "transport_error": transport_error,
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


def _distance_bin(distance: int | None) -> str:
    if distance is None:
        return "no_physical_terminal"
    if distance <= 2:
        return str(distance)
    if distance <= 4:
        return "3-4"
    if distance <= 9:
        return "5-9"
    return "10+"


def summarize_channel(
    rows: list[dict], key: str, *, gamma: float
) -> dict[str, object]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    targets = np.asarray(
        [float(row["target_discount"]) for row in rows], dtype=np.float64
    )
    terminals = np.asarray([bool(row["terminal"]) for row in rows])
    predicted_terminal = values < float(gamma) / 2.0
    has_terminal = bool(terminals.any())
    has_nonterminal = bool((~terminals).any())
    terminal_recall = (
        float(predicted_terminal[terminals].mean()) if has_terminal else None
    )
    nonterminal_recall = (
        float((~predicted_terminal[~terminals]).mean())
        if has_nonterminal
        else None
    )
    balanced_accuracy = (
        (terminal_recall + nonterminal_recall) / 2.0
        if terminal_recall is not None and nonterminal_recall is not None
        else None
    )
    return {
        "effective_discount_mean": float(values.mean()),
        "terminal_mean": (
            float(values[terminals].mean()) if has_terminal else None
        ),
        "nonterminal_mean": (
            float(values[~terminals].mean()) if has_nonterminal else None
        ),
        "brier": float(np.mean((values - targets) ** 2)),
        "failure_roc_auc": binary_roc_auc(
            (1.0 - values).tolist(), terminals.tolist()
        ),
        "balanced_accuracy_at_half_discount": balanced_accuracy,
    }


def summarize_rows(rows: list[dict], *, gamma: float) -> dict:
    terminal_rows = [row for row in rows if row["terminal"]]
    residuals = [
        abs(
            float(row["prior_error"])
            - float(row["posterior_error"])
            - float(row["transport_error"])
        )
        for row in rows
    ]
    distance_summary = {}
    for name in ("0", "1", "2", "3-4", "5-9", "10+", "no_physical_terminal"):
        selected = [row for row in rows if row["distance_bin"] == name]
        if selected:
            distance_summary[name] = {
                "count": len(selected),
                "posterior_discount_mean": float(
                    np.mean([row["posterior_discount"] for row in selected])
                ),
                "prior_discount_mean": float(
                    np.mean([row["prior_discount"] for row in selected])
                ),
                "kl_q_p_mean": float(
                    np.mean([row["kl_q_p"] for row in selected])
                ),
            }
    return {
        "transitions": len(rows),
        "terminal_transitions": len(terminal_rows),
        "posterior_expected": summarize_channel(
            rows, "posterior_discount", gamma=gamma
        ),
        "posterior_mode": summarize_channel(
            rows, "posterior_mode_discount", gamma=gamma
        ),
        "prior_expected": summarize_channel(rows, "prior_discount", gamma=gamma),
        "all_transition_errors": {
            key: _stats([float(row[key]) for row in rows])
            for key in ("prior_error", "posterior_error", "transport_error")
        },
        "terminal_transition_errors": {
            key: _stats([float(row[key]) for row in terminal_rows])
            for key in ("prior_error", "posterior_error", "transport_error")
        },
        "kl_q_p": _stats([float(row["kl_q_p"]) for row in rows]),
        "kl_q_p_terminal": _stats(
            [float(row["kl_q_p"]) for row in terminal_rows]
        ),
        "prior_probability_of_posterior_mode": _stats(
            [float(row["prior_probability_of_posterior_mode"]) for row in rows]
        ),
        "prior_probability_of_posterior_mode_terminal": _stats(
            [
                float(row["prior_probability_of_posterior_mode"])
                for row in terminal_rows
            ]
        ),
        "distance_to_terminal": distance_summary,
        "decomposition_max_abs_residual": max(residuals, default=0.0),
    }


def _sample_latents(
    probs: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    expanded = probs.expand(samples, *probs.shape[1:])
    flat = expanded.reshape(-1, expanded.shape[-1])
    indices = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    indices = indices.view(samples, probs.shape[-2])
    return F.one_hot(indices, num_classes=probs.shape[-1]).float()


def _continuation_for_samples(
    world_model,
    h: torch.Tensor,
    z_samples: torch.Tensor,
    imagination_discount: float,
) -> float:
    h_expanded = h.expand(z_samples.shape[0], -1)
    h_z = world_model.join_h_and_z(h_expanded, z_samples)
    prediction = torch.sigmoid(
        world_model.continue_predictor(h_z).squeeze(-1)
    )
    return float((imagination_discount * prediction).mean().item())


@torch.no_grad()
def collect_rows(
    cfg,
    actor,
    encoder,
    world_model,
    *,
    device: str,
    episodes: int,
    seed: int,
    samples: int,
) -> tuple[list[dict], list[float], int]:
    """Collect matched posterior/prior continuation predictions."""
    env = gym.make(cfg.environment_name)
    generator = torch.Generator(device=device).manual_seed(seed + 7_000_000)
    imagination_discount = learned_continue_discount(
        cfg.gamma, bool(getattr(cfg, "contdisc", True))
    )
    h_prev_backup = world_model.h_prev.clone()
    rows: list[dict] = []
    episode_returns: list[float] = []
    truncated_episodes = 0

    def posterior_from_observation(
        observation: np.ndarray,
        h_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = symlog(
            torch.from_numpy(np.asarray(observation, dtype=np.float32))
            .to(device)
            .unsqueeze(0)
        )
        tokens = encoder(state)
        logits = unimix_logits(
            world_model.compute_posterior(h_state, tokens), unimix_ratio=0.01
        )
        probs = F.softmax(logits, dim=-1)
        indices = probs.argmax(dim=-1)
        z_mode = F.one_hot(indices, num_classes=world_model.n_classes).float()
        return probs, z_mode, world_model.z_embedding(z_mode.view(1, -1))

    try:
        for episode_index in range(episodes):
            obs, _ = env.reset(seed=seed + episode_index)
            h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
            initial_z = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_embed = world_model.z_embedding(initial_z.view(1, -1))
            zero_action = torch.zeros(1, cfg.n_actions, device=device)
            h, _ = world_model.step_dynamics(z_embed, zero_action, h)
            _initial_probs, z_mode, z_embed = posterior_from_observation(obs, h)
            h_z = world_model.join_h_and_z(h, z_mode)
            episode_rows: list[dict] = []
            episode_return = 0.0

            while True:
                action = actor(h_z).argmax(dim=-1)
                action_onehot = F.one_hot(
                    action, num_classes=cfg.n_actions
                ).float()
                next_obs, reward, terminated, truncated, _ = env.step(
                    int(action.item())
                )
                episode_return += float(reward)

                h_next, prior_logits = world_model.step_dynamics(
                    z_embed, action_onehot, h
                )
                prior_probs = F.softmax(
                    unimix_logits(prior_logits, unimix_ratio=0.01), dim=-1
                )
                posterior_probs, posterior_mode, posterior_z_embed = (
                    posterior_from_observation(next_obs, h_next)
                )
                prior_samples = _sample_latents(
                    prior_probs, samples, generator
                )
                posterior_samples = _sample_latents(
                    posterior_probs, samples, generator
                )
                prior_discount = _continuation_for_samples(
                    world_model,
                    h_next,
                    prior_samples,
                    imagination_discount,
                )
                posterior_discount = _continuation_for_samples(
                    world_model,
                    h_next,
                    posterior_samples,
                    imagination_discount,
                )
                posterior_mode_h_z = world_model.join_h_and_z(
                    h_next, posterior_mode
                )
                posterior_mode_discount = float(
                    (
                        imagination_discount
                        * torch.sigmoid(
                            world_model.continue_predictor(
                                posterior_mode_h_z
                            ).squeeze()
                        )
                    ).item()
                )
                target_discount = 0.0 if terminated else float(cfg.gamma)
                log_q = torch.log(posterior_probs.clamp_min(1e-8))
                log_p = torch.log(prior_probs.clamp_min(1e-8))
                kl_q_p = float(
                    (posterior_probs * (log_q - log_p)).sum().item()
                )
                posterior_indices = posterior_probs.argmax(dim=-1, keepdim=True)
                modal_probability = float(
                    prior_probs.gather(-1, posterior_indices).mean().item()
                )
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "t": len(episode_rows),
                        "action": int(action.item()),
                        "terminal": bool(terminated),
                        "truncated": bool(truncated),
                        "target_discount": target_discount,
                        "posterior_discount": posterior_discount,
                        "posterior_mode_discount": posterior_mode_discount,
                        "prior_discount": prior_discount,
                        "kl_q_p": kl_q_p,
                        "prior_probability_of_posterior_mode": modal_probability,
                        **continuation_error_components(
                            prior_discount,
                            posterior_discount,
                            target_discount,
                        ),
                    }
                )

                if terminated or truncated:
                    truncated_episodes += int(bool(truncated))
                    break
                h = h_next
                z_mode = posterior_mode
                z_embed = posterior_z_embed
                h_z = posterior_mode_h_z

            physical_terminal = bool(episode_rows[-1]["terminal"])
            for index, row in enumerate(episode_rows):
                distance = len(episode_rows) - 1 - index if physical_terminal else None
                row["steps_to_terminal"] = distance
                row["distance_bin"] = _distance_bin(distance)
            rows.extend(episode_rows)
            episode_returns.append(episode_return)
    finally:
        env.close()
        world_model.h_prev = h_prev_backup

    return rows, episode_returns, truncated_episodes


@torch.no_grad()
def run_probe(
    checkpoint_path: Path,
    out_dir: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
    samples: int,
) -> dict:
    cfg, actor, _critic, _q_critic, encoder, world_model, checkpoint, _critic_key, _q_key = (
        load_checkpoint_models(checkpoint_path, device)
    )
    rows, returns, truncated_episodes = collect_rows(
        cfg,
        actor,
        encoder,
        world_model,
        device=device,
        episodes=episodes,
        seed=seed,
        samples=samples,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "train_step": checkpoint.get("step", checkpoint.get("train_step")),
        "device": device,
        "seed_start": seed,
        "episodes": episodes,
        "prior_and_posterior_samples": samples,
        "mean_episode_return": float(np.mean(returns)),
        "truncated_episodes": truncated_episodes,
        **summarize_rows(rows, gamma=float(cfg.gamma)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "rows.json").write_text(json.dumps(rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    if args.episodes <= 0 or args.samples < 2:
        parser.error("episodes must be positive and samples must be at least 2")

    device = resolve_device(args.device)
    summaries = []
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.resolve()
        name = f"{checkpoint.parent.parent.name}_{checkpoint.stem}"
        print(f"Probing continuation for {name} on {device}...")
        summary = run_probe(
            checkpoint,
            args.out / name,
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            samples=args.samples,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
