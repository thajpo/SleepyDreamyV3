"""Evaluate a trained policy on the environment."""
import argparse
import torch
import torch.nn.functional as F
import yaml
import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation

from .config import config
from .trainer_utils import initialize_actor, initialize_world_model


def load_env_config(config_path):
    """Load YAML config and apply overrides to global config."""
    with open(config_path, 'r') as f:
        overrides = yaml.safe_load(f)

    if 'environment' in overrides:
        for key, value in overrides['environment'].items():
            if hasattr(config.environment, key):
                setattr(config.environment, key, value)

    if 'models' in overrides:
        for key, value in overrides['models'].items():
            if hasattr(config.models, key):
                setattr(config.models, key, value)

    if 'train' in overrides:
        for key, value in overrides['train'].items():
            if hasattr(config.train, key):
                setattr(config.train, key, value)


def create_env(env_name):
    """Create environment with vision observations."""
    base_env = gym.make(env_name, render_mode="rgb_array")
    env = AddRenderObservation(
        base_env, render_only=False, render_key="pixels", obs_key="state"
    )
    return env


def evaluate(checkpoint_path, config_path, num_episodes=10, device="cuda"):
    """Run evaluation episodes and report statistics."""
    # Load config
    if config_path:
        load_env_config(config_path)

    print(f"Environment: {config.environment.environment_name}")
    print(f"Actions: {config.environment.n_actions}, Observations: {config.environment.n_observations}")
    print(f"d_hidden: {config.models.d_hidden}")
    print(f"Device: {device}")
    print()

    # Create environment
    env = create_env(config.environment.environment_name)
    n_actions = config.environment.n_actions

    # Initialize models
    actor = initialize_actor(device, cfg=config)
    encoder, world_model = initialize_world_model(device, batch_size=1, cfg=config)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle compiled model state dicts (have _orig_mod prefix)
    actor_state = checkpoint.get('actor', {})
    encoder_state = checkpoint.get('encoder', {})
    wm_state = checkpoint.get('world_model', {})

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
        h = torch.zeros(1, config.models.d_hidden * config.models.rnn.n_blocks, device=device)
        action_onehot = torch.zeros(1, n_actions, device=device)

        total_reward = 0
        steps = 0

        while True:
            # Prepare observation
            pixel_obs_t = torch.from_numpy(obs['pixels']).to(device).float().permute(2, 0, 1).unsqueeze(0)
            vec_obs_t = torch.from_numpy(obs['state']).to(device).float().unsqueeze(0)
            current_obs_dict = {"pixels": pixel_obs_t, "state": vec_obs_t}

            with torch.no_grad():
                # Encode observation
                posterior_logits = encoder(current_obs_dict)
                posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
                z_indices = posterior_dist.sample()
                z_onehot = F.one_hot(z_indices, num_classes=config.models.d_hidden // 16).float()
                z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())
                z_flat = z_sample.view(1, -1)

                # Step dynamics
                z_embed = world_model.z_embedding(z_flat)
                h, _ = world_model.step_dynamics(z_embed, action_onehot, h)

                # Get action from actor
                actor_input = world_model.join_h_and_z(h, z_sample)
                action_dist = torch.distributions.Categorical(logits=actor(actor_input))
                action = action_dist.sample()

            action_np = action.item()
            action_onehot = F.one_hot(action, num_classes=n_actions).float()

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: length={steps}, reward={total_reward:.1f}")

    env.close()

    # Print summary
    print()
    print("=" * 40)
    print(f"Results over {num_episodes} episodes:")
    print(f"  Avg length: {sum(episode_lengths)/len(episode_lengths):.1f}")
    print(f"  Avg reward: {sum(episode_rewards)/len(episode_rewards):.1f}")
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
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, help='Path to env config YAML')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.episodes, args.device)


if __name__ == "__main__":
    main()
