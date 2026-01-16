import gymnasium as gym
import numpy as np
import os
import time
from gymnasium.wrappers import AddRenderObservation
import h5py
import torch
import torch.nn.functional as F
from queue import Empty
import cv2
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from .config import config
from .trainer_utils import initialize_actor, initialize_world_model


def resize_image(img, target_size=(64, 64)):
    """Resize image using cv2 (much faster than torch on CPU)."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def create_env_with_vision(env_name, use_pixels=True):
    if use_pixels:
        base_env = gym.make(env_name, render_mode="rgb_array")
        env = AddRenderObservation(
            base_env, render_only=False, render_key="pixels", obs_key="state"
        )
    else:
        # State-only mode: no rendering overhead
        env = gym.make(env_name)
    return env

def collect_experiences(data_queue, model_queue, config, stop_event, log_dir=None):
    """
    Continuously collects experiences from the environment and puts them on a queue.

    Starts with random actions (fast, no model inference).
    Switches to learned policy when trainer sends first model update.
    Stops when stop_event is set by the trainer.
    """
    use_pixels = config.general.use_pixels
    env = create_env_with_vision(config.environment.environment_name, use_pixels=use_pixels)
    device = "cpu"
    n_actions = config.environment.n_actions

    # Start in random action mode - models initialized lazily
    use_random_actions = True
    actor = None
    encoder = None
    world_model = None

    episode_count = 0
    profiler = None
    profile_chunk_steps = 200
    if getattr(config.general, "profile", False) and log_dir:
        profile_dir = os.path.join(log_dir, "profiler", f"collector_{os.getpid()}")
        os.makedirs(profile_dir, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        trace_handler = tensorboard_trace_handler(profile_dir)
        profiler = profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=0, active=profile_chunk_steps, repeat=0),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        profiler.__enter__()
        print(
            "Collector profiler enabled for full run; saving a trace every "
            f"{profile_chunk_steps} steps. Traces: {profile_dir}"
        )

    while not stop_event.is_set():
        # Check for model updates from trainer
        try:
            new_model_states = model_queue.get_nowait()
            # First update: initialize models
            if actor is None:
                actor = initialize_actor(device=device, cfg=config)
                encoder, world_model = initialize_world_model(device, batch_size=1, cfg=config)
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

        while not stop_event.is_set():
            if use_random_actions:
                # Fast path: random action, no model inference
                action_np = env.action_space.sample()
                action_onehot_np = np.eye(n_actions, dtype=np.float32)[action_np]
            else:
                # Learned policy path
                if use_pixels:
                    # Pixel mode: resize and prepare dict input
                    resized_for_encoder = resize_image(obs['pixels'], target_size=(64, 64))
                    pixel_obs_t = torch.from_numpy(resized_for_encoder).to(device).float().permute(2, 0, 1).unsqueeze(0)
                    vec_obs_t = torch.from_numpy(obs['state']).to(device).float().unsqueeze(0)
                    encoder_input = {"pixels": pixel_obs_t, "state": vec_obs_t}
                else:
                    # State-only mode: encoder takes state tensor directly
                    vec_obs_t = torch.from_numpy(obs).to(device).float().unsqueeze(0)
                    encoder_input = vec_obs_t

                with torch.no_grad():
                    posterior_logits = encoder(encoder_input)
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

            # Store observations
            if use_pixels:
                # Resize pixels immediately (400x600 -> 64x64) to reduce memory/queue overhead
                resized_pixels = resize_image(obs["pixels"], target_size=(64, 64))
                episode_pixels.append(resized_pixels)
                episode_vec_obs.append(obs["state"])
            else:
                # State-only mode: obs is the state array directly
                episode_vec_obs.append(obs)

            episode_actions.append(action_onehot_np)
            episode_rewards.append(reward)
            episode_terminated.append(terminated)

            if profiler:
                profiler.step()

            if terminated or truncated:
                break

        # Only send complete episodes (not interrupted mid-episode by stop signal)
        if not stop_event.is_set() or (terminated or truncated):
            # Package and send episode
            if use_pixels:
                pixels_np = np.array(episode_pixels, dtype=np.uint8)
            else:
                pixels_np = None  # State-only mode: no pixels
            vec_obs_np = np.array(episode_vec_obs, dtype=np.float32)
            actions_np = np.array(episode_actions, dtype=np.float32)
            rewards_np = np.array(episode_rewards, dtype=np.float32)
            terminated_np = np.array(episode_terminated, dtype=bool)

            episode_length = len(vec_obs_np)
            data_queue.put((pixels_np, vec_obs_np, actions_np, rewards_np, terminated_np, episode_length))

            # Throttle if queue is getting full (backpressure)
            # This prevents wasteful over-collection in fast envs (e.g., state-only)
            queue_fill_ratio = data_queue.qsize() / data_queue._maxsize if data_queue._maxsize else 0
            if queue_fill_ratio > 0.8:
                time.sleep(0.05)  # Brief pause to let trainer catch up

    if profiler:
        profiler.__exit__(None, None, None)
    # Cleanup
    env.close()
    print("Experience collector: Stopped gracefully.")
