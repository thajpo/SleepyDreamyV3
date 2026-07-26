import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
from queue import Empty, Full

from ..models import initialize_actor, initialize_world_model, symlog, unimix_logits
from .env import create_env


logger = logging.getLogger(__name__)


def select_policy_action(action_logits: torch.Tensor, policy_mode: str) -> torch.Tensor:
    """Select a learned collector action under the configured behavior mode."""
    if policy_mode == "argmax":
        return action_logits.argmax(dim=-1)
    if policy_mode == "sample":
        return torch.distributions.Categorical(logits=action_logits).sample()
    raise ValueError(
        f"unsupported collector_policy_mode={policy_mode!r}; "
        "choose 'sample' or 'argmax'"
    )


def _queue_replay_packet(data_queue, packet, stop_event) -> bool:
    """Publish one bounded replay packet, respecting supervised shutdown."""
    while not stop_event.is_set():
        try:
            data_queue.put(packet, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _replay_arrays(
    pixels,
    states,
    actions,
    rewards,
    is_last,
    is_terminal,
    *,
    use_pixels: bool,
):
    """Freeze aligned collector lists into transport-owned NumPy arrays."""
    return (
        np.array(pixels, dtype=np.uint8) if use_pixels else None,
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(is_last, dtype=bool),
        np.array(is_terminal, dtype=bool),
    )


def collect_experiences(
    data_queue,
    model_queue,
    config,
    stop_event,
    log_dir=None,
    checkpoint_path=None,
    collector_id=0,
):
    """
    Continuously collects experiences from the environment and puts them on a queue.

    Starts with random actions (fast, no model inference).
    Switches to learned policy when trainer sends first model update.
    Stops when stop_event is set by the parent process supervisor.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(name)s %(message)s",
    )
    use_pixels = config.use_pixels
    collector_policy_mode = str(
        getattr(config, "collector_policy_mode", "sample")
    )
    if collector_policy_mode not in {"sample", "argmax"}:
        raise ValueError(
            f"unsupported collector_policy_mode={collector_policy_mode!r}; "
            "choose 'sample' or 'argmax'"
        )
    env = create_env(config.environment_name, use_pixels=use_pixels, config=config)
    device = "cpu"
    n_actions = config.n_actions
    action_repeat = getattr(config, "action_repeat", 1)
    base_seed = int(getattr(config, "seed", 0)) + 1000 + int(collector_id) * 10000
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32 - 1))
    torch.manual_seed(base_seed)
    if hasattr(env, "action_space"):
        env.action_space.seed(base_seed)

    # Fresh runs begin random; resumed runs can warm start from checkpoint.
    use_random_actions = checkpoint_path is None
    actor = None
    encoder = None
    world_model = None

    def initialize_models():
        nonlocal actor, encoder, world_model
        if actor is None:
            actor = initialize_actor(device=device, cfg=config)
            encoder, world_model = initialize_world_model(
                device, batch_size=1, cfg=config
            )

    if checkpoint_path is not None:
        initialize_models()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        actor.load_state_dict(checkpoint["actor"])
        encoder.load_state_dict(checkpoint["encoder"])
        world_model.load_state_dict(checkpoint["world_model"], strict=False)
        actor.eval()
        encoder.eval()
        world_model.eval()
        logger.info(
            "model_weights_loaded collector_id=%d source=checkpoint", collector_id
        )

    def pull_latest_models(episode_number: int) -> None:
        """Apply one coherent model snapshot at an episode boundary."""
        nonlocal actor, encoder, world_model, use_random_actions
        try:
            latest_model_update = model_queue.get_nowait()
        except Empty:
            return

        initialize_models()

        actor.load_state_dict(latest_model_update["actor"])
        encoder.load_state_dict(latest_model_update["encoder"])
        # strict=False: ignore h_prev/z_prev buffer shape mismatch (batch size differs)
        world_model.load_state_dict(latest_model_update["world_model"], strict=False)
        actor.eval()
        encoder.eval()
        world_model.eval()

        if use_random_actions:
            logger.info(
                "model_weights_loaded collector_id=%d version=%s episode=%d "
                "policy_mode=learned",
                collector_id,
                latest_model_update["version"],
                episode_number,
            )
            use_random_actions = False
        else:
            logger.info(
                "model_weights_loaded collector_id=%d version=%s episode=%d",
                collector_id,
                latest_model_update["version"],
                episode_number,
            )

    episode_count = 0
    continuous_delivery = bool(
        getattr(config, "continuous_replay_delivery", False)
    )
    replay_chunk_rows = int(config.sequence_length)

    while not stop_event.is_set():
        episode_count += 1
        # Recurrent carry is parameter-dependent. Swap encoder/RSSM/actor
        # snapshots only where the environment and carry are both reset, never
        # in the middle of an episode.
        pull_latest_models(episode_count)
        obs, info = env.reset(seed=base_seed + episode_count)

        (
            episode_pixels,
            episode_vec_obs,
            episode_actions,
            episode_rewards,
            episode_is_last,
            episode_is_terminal,
        ) = ([], [], [], [], [], [])

        if getattr(config, "replay_row_alignment", "post_action") == "reference":
            # Reference replay contains the environment reset observation. Its
            # aligned model input is the zero previous action and its reward is
            # zero because no transition has occurred yet.
            if use_pixels:
                episode_pixels.append(obs["pixels"])
                episode_vec_obs.append(np.zeros(1, dtype=np.float32))
            else:
                episode_vec_obs.append(obs)
            episode_actions.append(np.zeros(n_actions, dtype=np.float32))
            episode_rewards.append(0.0)
            episode_is_last.append(False)
            episode_is_terminal.append(False)

        h = None
        action_onehot = None
        z_prev_embed = None
        terminated = False
        truncated = False

        # Initialize world model state for learned policy
        if not use_random_actions:
            h = torch.zeros(1, config.d_hidden * config.rnn_n_blocks, device=device)
            action_onehot = torch.zeros(1, n_actions, device=device)
            z_prev = torch.zeros(
                1, world_model.n_latents, world_model.n_classes, device=device
            )
            z_prev_embed = world_model.z_embedding(z_prev.view(1, -1))
            logger.debug(
                "episode_started collector_id=%d episode=%d policy_mode=learned",
                collector_id,
                episode_count,
            )
        else:
            logger.debug(
                "episode_started collector_id=%d episode=%d policy_mode=random",
                collector_id,
                episode_count,
            )

        env_steps_in_episode = 0
        chunk_env_steps = 0
        chunk_row_offset = 0

        while not stop_event.is_set():
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
                    action_logits = unimix_logits(
                        action_logits,
                        unimix_ratio=float(getattr(config, "actor_unimix", 0.01)),
                    )
                    action = select_policy_action(
                        action_logits, collector_policy_mode
                    )

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
                chunk_env_steps += 1
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

            episode_complete = terminated or truncated
            if continuous_delivery and (
                len(episode_vec_obs) >= replay_chunk_rows or episode_complete
            ):
                arrays = _replay_arrays(
                    episode_pixels,
                    episode_vec_obs,
                    episode_actions,
                    episode_rewards,
                    episode_is_last,
                    episode_is_terminal,
                    use_pixels=use_pixels,
                )
                packet = (
                    *arrays,
                    chunk_env_steps,
                    collector_id,
                    episode_count,
                    episode_complete,
                    chunk_row_offset,
                )
                if not _queue_replay_packet(data_queue, packet, stop_event):
                    break
                chunk_row_offset += len(episode_vec_obs)
                chunk_env_steps = 0
                (
                    episode_pixels,
                    episode_vec_obs,
                    episode_actions,
                    episode_rewards,
                    episode_is_last,
                    episode_is_terminal,
                ) = ([], [], [], [], [], [])

            if episode_complete:
                break

        # Once training is complete, replay is no longer being consumed. Drop an
        # interrupted/final episode rather than blocking shutdown on a full queue.
        if not stop_event.is_set() and not continuous_delivery:
            # Package and send episode
            arrays = _replay_arrays(
                episode_pixels,
                episode_vec_obs,
                episode_actions,
                episode_rewards,
                episode_is_last,
                episode_is_terminal,
                use_pixels=use_pixels,
            )

            episode_length = env_steps_in_episode
            episode = (
                *arrays,
                episode_length,
                collector_id,
                episode_count,
            )
            _queue_replay_packet(data_queue, episode, stop_event)

        if not stop_event.is_set():
            # On-demand collection: wait if queue is sufficiently full.
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
    logger.info("collector_stopped collector_id=%d", collector_id)
