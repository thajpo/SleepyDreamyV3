#!/usr/bin/env python3
"""Evaluate CartPole checkpoints on one fixed set of reset seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import symlog, unimix_logits

if __package__:
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_q import load_checkpoint_models  # type: ignore[import-not-found]


def summarize_returns(returns: list[float]) -> dict[str, float | int]:
    """Summarize a non-empty set of CartPole episode returns."""
    if not returns:
        raise ValueError("at least one episode return is required")
    return {
        "episodes": len(returns),
        "mean_return": statistics.fmean(returns),
        "median_return": statistics.median(returns),
        "min_return": min(returns),
        "max_return": max(returns),
        "solved_fraction": sum(value >= 500.0 for value in returns) / len(returns),
    }


def select_policy_action(
    logits: torch.Tensor,
    *,
    policy_mode: str,
    generator: torch.Generator | None = None,
    actor_unimix: float = 0.01,
) -> torch.Tensor:
    """Select a deterministic or reproducibly sampled categorical action."""
    logits = unimix_logits(logits, unimix_ratio=actor_unimix)
    if policy_mode == "argmax":
        return logits.argmax(dim=-1)
    if policy_mode == "sample":
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
    raise ValueError(f"unsupported policy mode: {policy_mode}")


def select_posterior_latent(
    logits: torch.Tensor,
    *,
    latent_mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select posterior categories using evaluation or collection semantics."""
    mixed = unimix_logits(logits, unimix_ratio=0.01)
    if latent_mode == "argmax":
        return mixed.argmax(dim=-1)
    if latent_mode == "sample":
        probabilities = F.softmax(mixed, dim=-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        indices = torch.multinomial(flat, 1, generator=generator)
        return indices.view(*probabilities.shape[:-1])
    raise ValueError(f"unsupported latent mode: {latent_mode}")


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
    episodes: int,
    seed: int,
    policy_mode: str = "argmax",
    latent_mode: str = "argmax",
) -> dict[str, object]:
    """Run one policy mode without counterfactual probe overhead."""
    (
        cfg,
        actor,
        _critic,
        _q_critic,
        encoder,
        world_model,
        checkpoint,
        _critic_key,
        _q_key,
    ) = load_checkpoint_models(checkpoint_path, device)
    train_step = checkpoint.get("step", checkpoint.get("train_step"))
    env = gym.make(cfg.environment_name)
    env_spec = env.spec
    max_episode_steps = (
        int(env_spec.max_episode_steps)
        if env_spec is not None and env_spec.max_episode_steps is not None
        else 500
    )
    returns: list[float] = []
    h_prev_backup = world_model.h_prev.clone()
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(seed + 1_000_000)
    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(seed + 2_000_000)

    try:
        with torch.no_grad():
            for episode in range(episodes):
                obs, _ = env.reset(seed=seed + episode)
                h = torch.zeros(
                    1,
                    cfg.d_hidden * cfg.rnn_n_blocks,
                    device=device,
                )
                previous_action = torch.zeros(1, cfg.n_actions, device=device)
                z_prev = torch.zeros(
                    1,
                    world_model.n_latents,
                    world_model.n_classes,
                    device=device,
                )
                z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
                episode_return = 0.0

                for _timestep in range(max_episode_steps):
                    state = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    h, _ = world_model.step_dynamics(
                        z_prev_embed, previous_action, h
                    )
                    tokens = encoder(symlog(state.unsqueeze(0)))
                    posterior_logits = world_model.compute_posterior(h, tokens)
                    posterior_index = select_posterior_latent(
                        posterior_logits,
                        latent_mode=latent_mode,
                        generator=latent_generator,
                    )
                    z_sample = F.one_hot(
                        posterior_index,
                        num_classes=world_model.n_classes,
                    ).float()
                    actor_input = world_model.join_h_and_z(h, z_sample)
                    action = select_policy_action(
                        actor(actor_input),
                        policy_mode=policy_mode,
                        generator=action_generator,
                        actor_unimix=float(getattr(cfg, "actor_unimix", 0.01)),
                    )
                    previous_action = F.one_hot(
                        action, num_classes=cfg.n_actions
                    ).float()
                    z_prev_embed = world_model.z_embedding(z_sample.view(1, -1))

                    obs, reward, terminated, truncated, _ = env.step(
                        int(action.item())
                    )
                    episode_return += float(reward)
                    if terminated or truncated:
                        break
                returns.append(episode_return)
    finally:
        world_model.h_prev = h_prev_backup
        env.close()

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "train_step": train_step,
        "policy_mode": policy_mode,
        "latent_mode": latent_mode,
        "action_seed": seed + 1_000_000,
        "latent_seed": seed + 2_000_000,
        "seed_start": seed,
        "seed_end": seed + episodes - 1,
        "returns": returns,
        **summarize_returns(returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--policy-mode",
        choices=("argmax", "sample"),
        default="argmax",
        help="Select mode actions or reproducibly sample categorical actions",
    )
    parser.add_argument(
        "--latent-mode",
        choices=("argmax", "sample"),
        default="argmax",
        help="Select posterior mode or reproducibly sample as collection does",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    device = resolve_device(args.device)
    results = []
    for checkpoint in args.checkpoints:
        print(f"Evaluating {checkpoint} on {device}...")
        result = evaluate_checkpoint(
            checkpoint.resolve(),
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            policy_mode=args.policy_mode,
            latent_mode=args.latent_mode,
        )
        results.append(result)
        print(
            f"  step={result['train_step']} mean={result['mean_return']:.2f} "
            f"range={result['min_return']:.0f}--{result['max_return']:.0f}"
        )

    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
