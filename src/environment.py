import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AddRenderObservation
import h5py
import torch
import torch.nn.functional as F
from queue import Empty

from .config import config
from .trainer_utils import initialize_actor, initialize_world_model


def create_env_with_vision():
    base_env = gym.make(config.environment.environment_name, render_mode="rgb_array")

    env = AddRenderObservation(
        base_env, render_only=False, render_key="pixels", obs_key="state"
    )
    return env

def collect_experiences(data_queue, model_queue):
    """
    Continuously collects experiences from the environment and puts them on a queue.

    Starts with random actions (fast, no model inference).
    Switches to learned policy when trainer sends first model update.
    """
    env = create_env_with_vision()
    device = "cpu"
    n_actions = config.environment.n_actions

    # Start in random action mode - models initialized lazily
    use_random_actions = True
    actor = None
    encoder = None
    world_model = None

    episode_count = 0

    while True:
        # Check for model updates from trainer
        try:
            new_model_states = model_queue.get_nowait()
            # First update: initialize models
            if actor is None:
                actor = initialize_actor(device=device)
                encoder, world_model = initialize_world_model(device, batch_size=1)
            actor.load_state_dict(new_model_states['actor'])
            encoder.load_state_dict(new_model_states['encoder'])
            # strict=False: ignore h_prev/z_prev buffer shape mismatch (batch size differs)
            world_model.load_state_dict(new_model_states['world_model'], strict=False)
            actor.eval()
            encoder.eval()
            world_model.eval()
            if use_random_actions:
                print("Collector: Received models, switching to learned policy.")
                use_random_actions = False
            else:
                print("Collector: Updated models from trainer.")
        except Empty:
            pass

        episode_count += 1
        obs, info = env.reset()

        episode_pixels, episode_vec_obs, episode_actions, episode_rewards, episode_terminated = (
            [], [], [], [], []
        )

        # Initialize world model state for learned policy
        if not use_random_actions:
            h = torch.zeros(1, config.models.d_hidden * config.models.rnn.n_blocks, device=device)
            action_onehot = torch.zeros(1, n_actions, device=device)
            print(f"Collecting episode {episode_count} (learned policy)...")
        else:
            print(f"Collecting episode {episode_count} (random)...")

        while True:
            if use_random_actions:
                # Fast path: random action, no model inference
                action_np = env.action_space.sample()
                action_onehot_np = np.eye(n_actions, dtype=np.float32)[action_np]
            else:
                # Learned policy path
                pixel_obs_t = torch.from_numpy(obs['pixels']).to(device).float().permute(2, 0, 1).unsqueeze(0)
                vec_obs_t = torch.from_numpy(obs['state']).to(device).float().unsqueeze(0)
                current_obs_dict = {"pixels": pixel_obs_t, "state": vec_obs_t}

                with torch.no_grad():
                    posterior_logits = encoder(current_obs_dict)
                    posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
                    z_indices = posterior_dist.sample()
                    z_onehot = F.one_hot(z_indices, num_classes=config.models.d_hidden // 16).float()
                    z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())
                    z_flat = z_sample.view(1, -1)

                    z_embed = world_model.z_embedding(z_flat)
                    h, _ = world_model.step_dynamics(z_embed, action_onehot, h)

                    actor_input = world_model.join_h_and_z(h, z_sample)
                    action_dist = torch.distributions.Categorical(logits=actor(actor_input))
                    action = action_dist.sample()

                action_np = action.item()
                action_onehot = F.one_hot(action, num_classes=n_actions).float()
                action_onehot_np = action_onehot.cpu().numpy().squeeze()

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action_np)

            episode_pixels.append(obs["pixels"])
            episode_vec_obs.append(obs["state"])
            episode_actions.append(action_onehot_np)
            episode_rewards.append(reward)
            episode_terminated.append(terminated)

            if terminated or truncated:
                break

        # Package and send episode
        pixels_np = np.array(episode_pixels, dtype=np.uint8)
        vec_obs_np = np.array(episode_vec_obs, dtype=np.float32)
        actions_np = np.array(episode_actions, dtype=np.float32)
        rewards_np = np.array(episode_rewards, dtype=np.float32)
        terminated_np = np.array(episode_terminated, dtype=bool)

        data_queue.put((pixels_np, vec_obs_np, actions_np, rewards_np, terminated_np))
