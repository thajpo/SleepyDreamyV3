import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from queue import Empty
import cv2

from .trainer import initialize_actor, initialize_world_model, ProfilerManager, symlog, unimix_logits
from .utils import create_env


def resize_image(img, target_size=(64, 64)):
    """Resize image using cv2 (much faster than torch on CPU)."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def collect_experiences(data_queue, model_queue, config, stop_event, log_dir=None, wait_for_models=False):
    """
    Continuously collects experiences from the environment and puts them on a queue.

    Starts with random actions (fast, no model inference).
    Switches to learned policy when trainer sends first model update.
    Stops when stop_event is set by the trainer.

    Args:
        wait_for_models: If True, block until models are received before starting collection.
                        Use this in dreamer mode to avoid collecting random episodes.
    """
    use_pixels = config.general.use_pixels
    env = create_env(config.environment.environment_name, use_pixels=use_pixels)
    device = "cpu"
    n_actions = config.environment.n_actions

    # Start in random action mode - models initialized lazily
    use_random_actions = True
    actor = None
    encoder = None
    world_model = None

    episode_count = 0
    profile_enabled = getattr(config.general, "profile", False) and log_dir is not None
    profiler = ProfilerManager(
        enabled=profile_enabled,
        log_dir=log_dir or "",
        component_name=f"collector_{os.getpid()}",
        chunk_steps=200,
    )
    profiler.__enter__()

    # If wait_for_models is True, block until we receive initial models
    if wait_for_models:
        print("Collector: Waiting for initial models from trainer...", flush=True)
        while not stop_event.is_set():
            try:
                new_model_states = model_queue.get(timeout=1.0)  # Block with timeout
                actor = initialize_actor(device=device, cfg=config)
                encoder, world_model = initialize_world_model(device, batch_size=1, cfg=config)
                actor.load_state_dict(new_model_states['actor'])
                encoder.load_state_dict(new_model_states['encoder'])
                world_model.load_state_dict(new_model_states['world_model'], strict=False)
                actor.eval()
                encoder.eval()
                world_model.eval()
                use_random_actions = False
                print("Collector: Received initial models, starting with learned policy.", flush=True)
                break
            except Empty:
                continue  # Keep waiting

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
                    vec_obs_t = symlog(vec_obs_t)
                    encoder_input = {"pixels": pixel_obs_t, "state": vec_obs_t}
                else:
                    # State-only mode: encoder takes state tensor directly
                    vec_obs_t = torch.from_numpy(obs).to(device).float().unsqueeze(0)
                    vec_obs_t = symlog(vec_obs_t)
                    encoder_input = vec_obs_t

                with torch.no_grad():
                    posterior_logits = encoder(encoder_input)
                    posterior_logits_mixed = unimix_logits(posterior_logits, unimix_ratio=0.01)
                    posterior_dist = torch.distributions.Categorical(logits=posterior_logits_mixed)
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

            # On-demand collection: wait if queue is sufficiently full
            # This prevents wasteful over-collection in fast envs (e.g., state-only CartPole)
            while not stop_event.is_set():
                queue_fill_ratio = data_queue.qsize() / data_queue._maxsize if data_queue._maxsize else 0
                if queue_fill_ratio < 0.8:
                    break  # Queue has room, collect more
                time.sleep(0.1)  # Wait for trainer to drain queue

    profiler.__exit__(None, None, None)
    # Cleanup
    env.close()
    print("Experience collector: Stopped gracefully.")
