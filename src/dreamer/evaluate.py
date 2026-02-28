"""Evaluate a trained policy on the environment."""

import argparse
import torch
import torch.nn.functional as F

from dreamer.config import default_config
from dreamer.models import initialize_actor, initialize_world_model, symlog
from dreamer.runtime.env import create_env


def evaluate(
    config, checkpoint_path: str, num_episodes: int = 10, device: str = "cuda"
):
    """Run evaluation episodes and report statistics."""
    use_pixels = config.use_pixels

    print(f"Environment: {config.environment_name}")
    print(f"Actions: {config.n_actions}, Observations: {config.n_observations}")
    print(f"d_hidden: {config.d_hidden}")
    print(f"use_pixels: {use_pixels}")
    print(f"Device: {device}")
    print()

    # Create environment
    env = create_env(config.environment_name, use_pixels=use_pixels, config=config)
    n_actions = config.n_actions

    # Initialize models
    actor = initialize_actor(device, cfg=config)
    encoder, world_model = initialize_world_model(device, batch_size=1, cfg=config)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle compiled model state dicts (have _orig_mod prefix)
    actor_state = checkpoint.get("actor", {})
    encoder_state = checkpoint.get("encoder", {})
    wm_state = checkpoint.get("world_model", {})

    actor.load_state_dict(actor_state)
    encoder.load_state_dict(encoder_state)
    world_model.load_state_dict(wm_state, strict=False)

    actor.eval()
    encoder.eval()
    world_model.eval()

    print(f"Checkpoint loaded (step {checkpoint.get('step', 'unknown')})")
    print()

    # Run episodes
    episode_lengths = []
    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()

        # Initialize hidden state
        h = torch.zeros(1, config.d_hidden * config.rnn_n_blocks, device=device)
        action_onehot = torch.zeros(1, n_actions, device=device)
        num_classes = config.d_hidden // 16
        z_flat = torch.zeros(1, config.num_latents * num_classes, device=device)

        total_reward = 0.0
        steps = 0

        while True:
            # Prepare observation
            if use_pixels:
                pixel_obs_t = torch.from_numpy(obs["pixels"]).to(device).float()
                pixel_obs_t = pixel_obs_t.permute(2, 0, 1).unsqueeze(0)
                vec_obs_t = (
                    torch.from_numpy(obs["state"]).to(device).float().unsqueeze(0)
                )
                vec_obs_t = symlog(vec_obs_t)
                current_obs = {"pixels": pixel_obs_t, "state": vec_obs_t}
            else:
                vec_obs_t = torch.from_numpy(obs).to(device).float().unsqueeze(0)
                vec_obs_t = symlog(vec_obs_t)
                current_obs = vec_obs_t

            with torch.no_grad():
                # Step GRU first with previous z and action
                z_embed = world_model.z_embedding(z_flat)
                h, _ = world_model.step_dynamics(z_embed, action_onehot, h)

                # Encode observation into tokens
                tokens = encoder(current_obs)

                # Posterior conditioned on h_t: q(z_t | h_t, tokens)
                posterior_logits = world_model.compute_posterior(h, tokens)
                posterior_probs = F.softmax(posterior_logits, dim=-1)
                posterior_dist = torch.distributions.Categorical(
                    probs=posterior_probs
                )
                z_indices = posterior_dist.sample()
                z_onehot = F.one_hot(
                    z_indices, num_classes=config.d_hidden // 16
                ).float()
                z_sample = z_onehot + (
                    posterior_probs - posterior_probs.detach()
                )
                z_flat = z_sample.view(1, -1)

                # Get action from actor
                actor_input = world_model.join_h_and_z(h, z_sample)
                action_dist = torch.distributions.Categorical(logits=actor(actor_input))
                action = action_dist.sample()

            action_np = action.item()
            action_onehot = F.one_hot(action, num_classes=n_actions).float()

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: length={steps}, reward={total_reward:.1f}")

    env.close()

    # Print summary
    print()
    print("=" * 40)
    print(f"Results over {num_episodes} episodes:")
    print(f"  Avg length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"  Avg reward: {sum(episode_rewards) / len(episode_rewards):.1f}")
    print(f"  Min length: {min(episode_lengths)}")
    print(f"  Max length: {max(episode_lengths)}")
    print("=" * 40)

    # CartPole success criteria
    avg_len = sum(episode_lengths) / len(episode_lengths)
    if avg_len >= 475:
        print("SOLVED! (avg >= 475)")
    elif avg_len >= 400:
        print("Good! (avg >= 400)")
    elif avg_len >= 200:
        print("Learning... (avg >= 200)")
    else:
        print("Still training needed (avg < 200)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DreamerV3 policy")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)"
    )
    args = parser.parse_args()

    cfg = default_config()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    evaluate(
        config=cfg,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        device=device,
    )


if __name__ == "__main__":
    main()
