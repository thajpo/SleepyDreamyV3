import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from queue import Empty

from ..models import initialize_actor, initialize_world_model, symlog, unimix_logits
from .env import create_env


def collect_experiences(data_queue, model_queue, config, stop_event, log_dir=None):
    """
    Continuously collects experiences from the environment and puts them on a queue.

    Starts with random actions (fast, no model inference).
    Switches to learned policy when trainer sends first model update.
    Stops when stop_event is set by the trainer.
    """
    use_pixels = config.use_pixels
    env = create_env(config.environment_name, use_pixels=use_pixels, config=config)
    device = "cpu"
    n_actions = config.n_actions
    action_repeat = getattr(config, "action_repeat", 1)

    # Start in random action mode - models initialized lazily
    use_random_actions = True
    actor = None
    encoder = None
    world_model = None

    def pull_latest_models() -> int:
        """Drain model queue and apply only the newest update."""
        nonlocal actor, encoder, world_model, use_random_actions
        latest_model_states = None
        updates = 0
        while True:
            try:
                latest_model_states = model_queue.get_nowait()
                updates += 1
            except Empty:
                break

        if latest_model_states is None:
            return 0

        if actor is None:
            actor = initialize_actor(device=device, cfg=config)
            encoder, world_model = initialize_world_model(
                device, batch_size=1, cfg=config
            )

        actor.load_state_dict(latest_model_states["actor"])
        encoder.load_state_dict(latest_model_states["encoder"])
        # strict=False: ignore h_prev/z_prev buffer shape mismatch (batch size differs)
        world_model.load_state_dict(latest_model_states["world_model"], strict=False)
        actor.eval()
        encoder.eval()
        world_model.eval()

        if use_random_actions:
            print("Collector: Received models, switching to learned policy.")
            use_random_actions = False
        else:
            print("Collector: Updated models from trainer.")
        return updates

    episode_count = 0

    while not stop_event.is_set():
        pull_latest_models()

        episode_count += 1
        obs, info = env.reset()

        (
            episode_pixels,
            episode_vec_obs,
            episode_actions,
            episode_rewards,
            episode_is_last,
            episode_is_terminal,
        ) = ([], [], [], [], [], [])

        h = None
        action_onehot = None
        z_prev_embed = None

        # Initialize world model state for learned policy
        if not use_random_actions:
            h = torch.zeros(1, config.d_hidden * config.rnn_n_blocks, device=device)
            action_onehot = torch.zeros(1, n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
            print(f"Collecting episode {episode_count} (learned policy)...")
        else:
            print(f"Collecting episode {episode_count} (random)...")

        env_steps_in_episode = 0

        while not stop_event.is_set():
            pull_latest_models()

            if use_random_actions:
                # Fast path: random action, no model inference
                action_np = env.action_space.sample()
                action_onehot_np = np.eye(n_actions, dtype=np.float32)[action_np]
            else:
                if h is None or action_onehot is None or z_prev_embed is None:
                    h = torch.zeros(
                        1, config.d_hidden * config.rnn_n_blocks, device=device
                    )
                    action_onehot = torch.zeros(1, n_actions, device=device)
                    z_prev = torch.zeros(
                        1, world_model.n_latents, world_model.n_classes, device=device
                    )
                    z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))

                # Learned policy path
                if use_pixels:
                    # Pixel mode: use environment-provided frame directly.
                    pixel_obs_t = (
                        torch.from_numpy(obs["pixels"])
                        .to(device)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    vec_obs_t = (
                        torch.from_numpy(obs["state"]).to(device).float().unsqueeze(0)
                    )
                    vec_obs_t = symlog(vec_obs_t)
                    encoder_input = {"pixels": pixel_obs_t, "state": vec_obs_t}
                else:
                    # State-only mode: encoder takes state tensor directly
                    vec_obs_t = torch.from_numpy(obs).to(device).float().unsqueeze(0)
                    vec_obs_t = symlog(vec_obs_t)
                    encoder_input = vec_obs_t

                with torch.no_grad():
                    h, _ = world_model.step_dynamics(z_prev_embed, action_onehot, h)

                    # Encoder now returns tokens, not logits
                    tokens = encoder(encoder_input)

                    # Posterior is conditioned on h_t: q(z_t | h_t, tokens)
                    posterior_logits = world_model.compute_posterior(h, tokens)
                    posterior_logits = unimix_logits(
                        posterior_logits, unimix_ratio=0.01
                    )
                    posterior_probs = F.softmax(posterior_logits, dim=-1)
                    posterior_dist = torch.distributions.Categorical(
                        probs=posterior_probs
                    )
                    z_indices = posterior_dist.sample()
                    num_classes = config.d_hidden // 16
                    z_onehot = F.one_hot(z_indices, num_classes=num_classes).float()
                    z_sample = z_onehot + (posterior_probs - posterior_probs.detach())

                    actor_input = world_model.join_h_and_z(h, z_sample)
                    action_logits = actor(actor_input)
                    action_logits = unimix_logits(action_logits, unimix_ratio=0.01)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()

                action_np = action.item()
                action_onehot = F.one_hot(action, num_classes=n_actions).float()
                z_prev_embed = world_model.z_embedding(z_sample.view(1, -1))
                action_onehot_np = action_onehot.cpu().numpy().squeeze()

            # Execute action with repeat
            total_reward = 0.0
            terminated = False
            truncated = False
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(action_np)
                total_reward += float(reward)
                env_steps_in_episode += 1
                if terminated or truncated:
                    break

            # Store observations
            if use_pixels:
                episode_pixels.append(obs["pixels"])
                # obs["state"] is the same as pixels for Atari - store dummy vector
                episode_vec_obs.append(np.zeros(1, dtype=np.float32))
            else:
                # State-only mode: obs is the state array directly
                episode_vec_obs.append(obs)

            episode_actions.append(action_onehot_np)
            episode_rewards.append(total_reward)
            episode_is_last.append(terminated or truncated)
            episode_is_terminal.append(terminated)

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
            is_last_np = np.array(episode_is_last, dtype=bool)
            is_terminal_np = np.array(episode_is_terminal, dtype=bool)

            episode_length = env_steps_in_episode
            data_queue.put(
                (
                    pixels_np,
                    vec_obs_np,
                    actions_np,
                    rewards_np,
                    is_last_np,
                    is_terminal_np,
                    episode_length,
                )
            )

            # On-demand collection: wait if queue is sufficiently full
            # This prevents wasteful over-collection in fast envs (e.g., state-only CartPole)
            while not stop_event.is_set():
                queue_fill_ratio = (
                    data_queue.qsize() / data_queue._maxsize
                    if data_queue._maxsize
                    else 0
                )
                if queue_fill_ratio < 0.8:
                    break  # Queue has room, collect more
                time.sleep(0.1)  # Wait for trainer to drain queue

    # Cleanup
    env.close()
    print("Experience collector: Stopped gracefully.")
