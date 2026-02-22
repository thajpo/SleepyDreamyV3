import math
import copy
import json
from dataclasses import asdict
import torch
import torch.distributions as dist
import torch.nn.functional as F
from queue import Full
import mlflow
import os
import time

from ..runtime.replay_buffer import EpisodeReplayBuffer
from ..models import (
    symlog,
    symexp,
    resize_pixels_to_target,
    unimix_logits,
    twohot_encode,
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    compute_wm_loss,
    compute_actor_critic_losses,
    dream_sequence,
    calculate_lambda_returns,
    LaProp,
    adaptive_gradient_clipping,
)
from .mlflow_logger import MLflowLogger
from ..runtime.env import create_env


class WorldModelTrainer:
    def __init__(
        self,
        config,
        data_queue,
        model_queue,
        log_dir,
        checkpoint_path=None,
        mlflow_run_id=None,
        dry_run=False,
    ):
        self.config = config  # Store config for use in methods
        self.dry_run = dry_run

        device_str = config.device
        if device_str == "auto":
            if torch.cuda.is_available():
                device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"
        self.device = torch.device(device_str)

        self.use_pixels = config.use_pixels
        self.n_dream_steps = config.num_dream_steps
        self.gamma = config.gamma
        self.lam = config.lam
        self.actor_entropy_coef = config.actor_entropy_coef
        b_start = config.b_start
        b_end = config.b_end
        beta_range = torch.arange(
            start=b_start,
            end=b_end,
            device=self.device,
        )
        self.B = symexp(beta_range)
        self.S = 0.0

        self.actor = initialize_actor(self.device, config)
        self.critic = initialize_critic(self.device, config)
        self.encoder, self.world_model = initialize_world_model(
            self.device, config, batch_size=config.batch_size
        )

        self.compile_models = getattr(config, "compile_models", False)
        if self.compile_models:
            if hasattr(torch, "compile"):
                try:
                    self.encoder = torch.compile(self.encoder)
                    self.world_model = torch.compile(self.world_model)
                    self.actor = torch.compile(self.actor)
                    self.critic = torch.compile(self.critic)
                    print("torch.compile enabled for trainer models.")
                except Exception as exc:
                    print(f"torch.compile failed ({exc}); running uncompiled.")
            else:
                print("torch.compile not available; running uncompiled.")

        self.wm_params = list(self.encoder.parameters()) + list(
            self.world_model.parameters()
        )
        # LaProp optimizer (DreamerV3): more stable than Adam for RL
        self.wm_optimizer = LaProp(
            self.wm_params,
            lr=config.wm_lr,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = LaProp(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )
        self.actor_optimizer = LaProp(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )

        # distinct critic target network for stability (DreamerV3)
        self.critic_ema = copy.deepcopy(self.critic)
        for param in self.critic_ema.parameters():
            param.requires_grad = False

        self.critic_ema_decay = getattr(config, "critic_ema_decay", 0.98)
        self.critic_ema_regularizer = getattr(config, "critic_ema_regularizer", 1.0)
        self.critic_replay_scale = getattr(config, "critic_replay_scale", 0.0)

        self.max_train_steps = config.max_train_steps
        self.train_step = 0
        self.data_queue = data_queue
        self.model_queue = model_queue
        self.d_hidden = config.d_hidden
        self.num_latents = config.num_latents  # L
        self.num_classes = config.d_hidden // 16  # K
        self.n_actions = config.n_actions
        self.steps_per_weight_sync = config.steps_per_weight_sync
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.replay_ratio = getattr(config, "replay_ratio", 1.0)
        self.action_repeat = getattr(config, "action_repeat", 1)
        self.profile_enabled = getattr(config, "profile", False)

        # Replay buffer: background thread drains queue, sample() returns instantly
        self.replay_buffer = EpisodeReplayBuffer(
            data_queue=data_queue,
            max_episodes=config.replay_buffer_size,
            min_episodes=config.min_buffer_episodes,
            sequence_length=config.sequence_length,
        )
        self.replay_buffer.start()
        print(f"Replay buffer started (max={config.replay_buffer_size} episodes)")

        # Initialize MLflow logger with the provided log directory
        self.mlflow_run_id = mlflow_run_id
        self.logger = MLflowLogger(log_dir=log_dir, run_id=mlflow_run_id)
        self.log_dir = log_dir
        print(f"MLflow logging to: {log_dir}")
        self.log_profile = getattr(config, "log_profile", "lean")
        if self.log_profile not in ("lean", "full"):
            print(
                f"Warning: unknown log_profile={self.log_profile}, falling back to lean"
            )
            self.log_profile = "lean"
        self.obs_mode = (
            "hybrid"
            if self.use_pixels and config.n_observations > 0
            else "vision"
            if self.use_pixels
            else "vector"
        )
        self._has_pixel_obs = self.use_pixels
        self._has_vector_obs = config.n_observations > 0
        print(f"Logging profile: {self.log_profile} | obs_mode: {self.obs_mode}")
        self.log_every = 250
        self.image_log_every = 2500
        self.surprise_ema_beta = config.surprise_ema_beta
        self._wm_surprise_ema = {}
        self._last_surprise_log_ratio = 0.0  # Raw surprise (log ratio)
        self._smoothed_surprise = (
            0.0  # Smoothed surprise for LR scaling (lingers after spikes)
        )
        self.surprise_smooth_beta = (
            0.9  # How fast smoothed surprise decays (higher = slower)
        )
        self.surprise_scale_ac_lr = config.surprise_scale_ac_lr
        self.surprise_lr_scale_k = config.surprise_lr_scale_k
        self.base_actor_lr = config.actor_lr
        self.base_critic_lr = config.critic_lr
        self._wm_surprise_eps = 1e-8

        # WM focus mode: extra WM steps when surprise spikes
        self.surprise_wm_focus_threshold = config.surprise_wm_focus_threshold
        self.surprise_wm_focus_ratio = config.surprise_wm_focus_ratio
        self.surprise_wm_focus_duration = config.surprise_wm_focus_duration
        self.surprise_wm_focus_cooldown = config.surprise_wm_focus_cooldown
        self._wm_focus_steps_remaining = 0  # Countdown for focus mode
        self._wm_focus_cooldown_remaining = 0  # Cooldown after focus (no re-entry)
        self._wm_focus_ac_counter = 0  # Counter for AC step ratio

        # WM:AC update ratio (e.g., 4 = do 4 WM updates per 1 AC update)
        self.wm_ac_ratio = config.wm_ac_ratio
        self._wm_ac_counter = 0  # Counts WM steps since last AC step

        # Cosine LR decay
        self.lr_cosine_decay = config.lr_cosine_decay
        self.lr_cosine_min_factor = config.lr_cosine_min_factor

        # Cosine WM:AC ratio schedule
        self.wm_ac_ratio_cosine = config.wm_ac_ratio_cosine
        self.wm_ac_ratio_max = config.wm_ac_ratio_max
        self.wm_ac_ratio_min = config.wm_ac_ratio_min
        self.wm_ac_ratio_invert = config.wm_ac_ratio_invert

        # Baseline mode: disable non-paper extras for baseline experiments
        self.baseline_mode = getattr(config, "baseline_mode", False)
        if self.baseline_mode:
            self.surprise_scale_ac_lr = False
            self.wm_ac_ratio_cosine = False
            self.lr_cosine_decay = False
            print(
                "BASELINE MODE: disabled surprise_scale_ac_lr, wm_ac_ratio_cosine, lr_cosine_decay"
            )

        # Deterministic evaluation
        self.eval_every = config.eval_every
        self.eval_episodes = config.eval_episodes
        self._ac_training_started = False  # Track if we've entered AC training phase

        # Early stopping (0 to disable)
        self.early_stop_ep_length = getattr(config, "early_stop_ep_length", 0)

        # Checkpointing
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_interval = config.checkpoint_interval

        # Save config snapshot
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Timing
        self.step_times = []
        self.train_start_time = time.time()
        self.last_log_time = time.time()
        self.steps_since_log = 0
        # Resume-aware replay gating baseline.
        # On fresh runs this stays 0. On resumed runs we seed a baseline so
        # replay_ratio gating does not stall waiting for impossible new env steps.
        self._resume_env_steps_offset = 0

        # Checkpoint loading
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            denom = max(1e-8, self.replay_ratio)
            self._resume_env_steps_offset = int(
                self.train_step
                * self.batch_size
                * self.sequence_length
                * self.action_repeat
                / denom
            )

    def get_data_from_buffer(self):
        """Sample batch from replay buffer (non-blocking after initial fill)."""
        # Sample returns list of (pixels, states, actions, rewards, is_last, is_terminal, mask) tuples
        # Each already has fixed sequence_length from buffer's _sample_subsequence
        batch = self.replay_buffer.sample(self.batch_size)

        batch_pixels = []
        batch_pixels_original = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_is_last = []
        batch_is_terminal = []
        batch_mask = []

        for pixels, states, actions, rewards, is_last, is_terminal, mask in batch:
            if self.use_pixels and pixels is not None:
                target_size = self.config.encoder_cnn_target_size
                # Convert pixels from (T, H, W, C) to (T, C, H, W)
                pixels_tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                batch_pixels_original.append(pixels_tensor)  # Keep original resolution
                if pixels_tensor.shape[-2:] != target_size:
                    pixels_tensor = resize_pixels_to_target(pixels_tensor, target_size)
                batch_pixels.append(pixels_tensor)

            batch_states.append(torch.from_numpy(states))
            batch_actions.append(torch.from_numpy(actions))
            batch_rewards.append(torch.from_numpy(rewards))
            batch_is_last.append(torch.from_numpy(is_last))
            batch_is_terminal.append(torch.from_numpy(is_terminal))
            batch_mask.append(torch.from_numpy(mask))

        # Stack into batch tensors: (B, T, ...)
        # Note: All sequences are same length now (buffer handles this)
        if self.use_pixels and batch_pixels:
            self.pixels = (
                torch.stack(batch_pixels).to(self.device).float()
            )  # (B, T, C, H, W)
            self.pixels_original = (
                torch.stack(batch_pixels_original).to(self.device).float()
            )
        else:
            self.pixels = None
            self.pixels_original = None

        self.states = symlog(
            torch.stack(batch_states).to(self.device)
        )  # (B, T, state_dim)
        self.actions = torch.stack(batch_actions).to(self.device)  # (B, T, n_actions)
        self.rewards = torch.stack(batch_rewards).to(self.device)  # (B, T)
        self.is_last = torch.stack(batch_is_last).to(self.device)  # (B, T)
        self.is_terminal = torch.stack(batch_is_terminal).to(self.device)  # (B, T)
        self.mask = torch.stack(batch_mask).to(self.device)  # (B, T) - 1=real, 0=padded

    def train_models(self):
        while self.train_step < self.max_train_steps:
            # Replay ratio gating: wait if we've trained too fast relative to env steps
            env_steps = (
                self.replay_buffer.total_env_steps + self._resume_env_steps_offset
            )
            target_train_steps = int(
                env_steps
                * self.replay_ratio
                / (self.batch_size * self.sequence_length * self.action_repeat)
            )
            if self.train_step >= target_train_steps and env_steps > 0:
                time.sleep(0.01)  # Brief wait for more data
                continue

            # Apply cosine LR schedule (if enabled)
            self.apply_lr_schedule()

            # Data loading
            self.get_data_from_buffer()  # Sample from replay buffer (non-blocking)

            # Skip if no data was retrieved
            if not hasattr(self, "states") or self.states.shape[1] == 0:
                print(f"Trainer: No data was retrieved at step {self.train_step}.")
                continue

            # Reset hidden states per trajectory - match actual input batch size
            actual_batch_size = self.states.shape[0]
            h_dim = self.world_model.h_prev.shape[1]
            self.world_model.h_prev = torch.zeros(
                actual_batch_size, h_dim, device=self.device
            )
            self.world_model.z_prev = torch.zeros(
                actual_batch_size,
                self.num_latents,
                self.num_classes,
                device=self.device,
            )

            # Zero gradients before accumulating losses
            self.wm_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            # Initialize loss accumulators - will be set on first iteration
            total_wm_loss = None
            total_actor_loss = None
            total_critic_loss = None
            # Accumulate individual loss components for logging (as tensors to avoid GPU syncs in loop)
            wm_loss_components = {
                "prediction_pixel": torch.tensor(0.0, device=self.device),
                "prediction_vector": torch.tensor(0.0, device=self.device),
                "prediction_reward": torch.tensor(0.0, device=self.device),
                "prediction_continue": torch.tensor(0.0, device=self.device),
                "dynamics": torch.tensor(0.0, device=self.device),
                "representation": torch.tensor(0.0, device=self.device),
                "kl_dynamics_raw": torch.tensor(0.0, device=self.device),
                "kl_representation_raw": torch.tensor(0.0, device=self.device),
            }
            # Accumulate dreamed trajectory stats for logging
            dreamed_rewards_list = []
            dreamed_values_list = []
            actor_entropy_list = []  # Track actor entropy for monitoring
            # Replay grounding accumulators (posterior states + value annotations)
            replay_posterior_states = []
            replay_value_annotations = []
            replay_loss = None
            replay_ema_reg = None
            # Initialize loss variables in case loop doesn't execute
            wm_loss = torch.tensor(0.0, device=self.device)
            actor_loss = torch.tensor(0.0, device=self.device)
            critic_loss = torch.tensor(0.0, device=self.device)
            last_obs_pixels = None
            last_obs_pixels_original = None
            last_reconstruction_pixels = None
            last_posterior_probs = None
            last_dreamed_pixels = None

            # --- PROFILE: Forward pass ---
            t0 = time.perf_counter()

            # Phase 2: Batch encode all timesteps at once (single encoder forward pass)
            B, T = self.states.shape[:2]
            burn_in_steps = min(
                getattr(self.config, "replay_burn_in", 8), max(0, T - 1)
            )
            train_start_t = burn_in_steps
            effective_train_steps = T - train_start_t
            states_flat = self.states.view(B * T, self.states.shape[-1])  # (B*T, n_obs)

            if self.use_pixels and self.pixels is not None:
                pixels_flat = self.pixels.view(
                    B * T, *self.pixels.shape[2:]
                )  # (B*T, C, H, W)
                encoder_input = {"pixels": pixels_flat, "state": states_flat}
            else:
                encoder_input = states_flat  # State-only mode

            all_posterior_logits = self.encoder(
                encoder_input
            )  # (B*T, d_hidden, categories)
            # Reshape back to (B, T, d_hidden, categories)
            all_posterior_logits = all_posterior_logits.view(
                B, T, *all_posterior_logits.shape[1:]
            )

            # Decide AC update once per batch (ratio + focus), then mask per timestep
            skip_ac_batch = False
            ac_any = False
            current_ratio = self.get_current_wm_ac_ratio()
            self._wm_ac_counter += 1
            if self._wm_ac_counter < current_ratio:
                skip_ac_batch = True
            else:
                self._wm_ac_counter = 0
                skip_ac_batch = self.should_skip_ac_for_wm_focus()

            for t_step in range(T):
                # Extract time step - now just indexing, no encoder call
                if self.use_pixels and self.pixels is not None:
                    obs_t = {
                        "pixels": self.pixels[:, t_step],
                        "state": self.states[:, t_step],
                    }
                else:
                    obs_t = {"state": self.states[:, t_step]}  # State-only mode
                action_t = self.actions[:, t_step]
                reward_t = self.rewards[:, t_step]
                is_terminal_t = self.is_terminal[:, t_step]
                sample_mask = self.mask[:, t_step]

                # Use pre-computed encoder output with unimix (DreamerV3 Section 4)
                posterior_logits = all_posterior_logits[:, t_step]
                posterior_logits_mixed = unimix_logits(
                    posterior_logits, unimix_ratio=0.01
                )
                posterior_dist = dist.Categorical(
                    logits=posterior_logits_mixed, validate_args=False
                )

                (
                    obs_reconstruction,
                    reward_dist,
                    continue_logits,
                    h_z_joined,
                    posterior_z_sample,
                    prior_logits,
                ) = self.world_model(posterior_dist, action_t)

                # Shape verification (DreamerV3 paper alignment)
                if t_step == 0:  # Check once per batch
                    L, K = self.num_latents, self.num_classes
                    h_dim = self.world_model.n_blocks * self.d_hidden
                    assert posterior_logits.shape[-2:] == (L, K), (
                        f"posterior_logits shape {posterior_logits.shape} != [B, {L}, {K}]"
                    )
                    assert prior_logits.shape[-2:] == (L, K), (
                        f"prior_logits shape {prior_logits.shape} != [B, {L}, {K}]"
                    )
                    assert h_z_joined.shape[-1] == h_dim + L * K, (
                        f"h_z_joined dim {h_z_joined.shape[-1]} != {h_dim + L * K}"
                    )

                # Updating loss of encoder and world model
                wm_loss, wm_loss_dict = compute_wm_loss(
                    obs_reconstruction,
                    obs_t,
                    reward_dist,
                    reward_t,
                    is_terminal_t,
                    continue_logits,
                    posterior_dist,
                    prior_logits,
                    self.B,
                    self.config,
                    self.device,
                    use_pixels=self.use_pixels,
                    sample_mask=sample_mask,
                )

                # Collect posterior states for replay grounding (detach to avoid WM grads)
                if self.critic_replay_scale > 0.0:
                    replay_posterior_states.append(h_z_joined.detach())

                # Accumulate individual loss components (tensor addition, no GPU sync)
                for key in wm_loss_components:
                    wm_loss_components[key] = (
                        wm_loss_components[key] + wm_loss_dict[key].detach()
                    )

                # Store visualization data
                if self.use_pixels and "pixels" in obs_t:
                    last_obs_pixels = obs_t["pixels"]
                    last_obs_pixels_original = self.pixels_original[:, t_step]
                    last_reconstruction_pixels = obs_reconstruction.get("pixels")
                else:
                    last_obs_pixels = None
                    last_obs_pixels_original = None
                    last_reconstruction_pixels = None
                last_posterior_probs = (
                    posterior_dist.probs.detach()
                )  # (batch, d_hidden, d_hidden/16)

                # --- Dream Sequence for Actor-Critic ---
                # Skip AC for this timestep if batch-level skip is active or no valid samples.
                if t_step < train_start_t:
                    continue

                valid_ac_step = sample_mask.sum() > 0
                if not skip_ac_batch and valid_ac_step:
                    h_prev_backup = self.world_model.h_prev.clone()
                    (
                        dreamed_recurrent_states,
                        dreamed_actions_logits,
                        dreamed_actions_sampled,
                    ) = dream_sequence(
                        h_z_joined,
                        self.world_model.z_embedding(
                            posterior_z_sample.view(actual_batch_size, -1)
                        ),
                        self.n_dream_steps,
                        self.actor,
                        self.world_model,
                        self.n_actions,
                        self.d_hidden,
                    )
                    self.world_model.h_prev = h_prev_backup

                    dreamed_rewards_logits = self.world_model.reward_predictor(
                        dreamed_recurrent_states
                    ).detach()
                    dreamed_rewards_probs = F.softmax(dreamed_rewards_logits, dim=-1)
                    dreamed_rewards = torch.sum(
                        dreamed_rewards_probs * self.B, dim=-1
                    ).detach()
                    dreamed_rewards_list.append(dreamed_rewards.detach().cpu())

                    dreamed_continues = (
                        self.world_model.continue_predictor(dreamed_recurrent_states)
                        .detach()
                        .squeeze(-1)
                    )  # Remove trailing (1,) dimension

                    dreamed_values_logits = self.critic(dreamed_recurrent_states)
                    with torch.no_grad():
                        dreamed_values_logits_ema = self.critic_ema(
                            dreamed_recurrent_states
                        )
                    dreamed_values_probs = F.softmax(dreamed_values_logits, dim=-1)
                    dreamed_values = torch.sum(dreamed_values_probs * self.B, dim=-1)
                    dreamed_values_list.append(dreamed_values.detach().cpu())

                    lambda_returns = calculate_lambda_returns(
                        dreamed_rewards,
                        dreamed_values,
                        dreamed_continues,
                        self.gamma,
                        self.lam,
                        self.n_dream_steps,
                    )

                    # Store per-timestep imagination return annotation for replay loss
                    if self.critic_replay_scale > 0.0:
                        replay_value_annotations.append(lambda_returns[0].detach())

                    self.update_return_scale(lambda_returns)

                    (
                        actor_loss,
                        critic_loss,
                        entropy,
                    ) = compute_actor_critic_losses(
                        dreamed_values_logits,
                        dreamed_values,
                        lambda_returns,
                        dreamed_continues,
                        dreamed_actions_logits,
                        dreamed_actions_sampled,
                        self.B,
                        self.S,
                        self.gamma,
                        actor_entropy_coef=self.actor_entropy_coef,
                        dreamed_values_logits_ema=dreamed_values_logits_ema,
                        critic_ema_coef=self.critic_ema_regularizer,
                        sample_mask=sample_mask,
                    )
                    actor_entropy_list.append(entropy.detach().cpu())
                    ac_any = True
                else:
                    actor_loss = torch.tensor(0.0, device=self.device)
                    critic_loss = torch.tensor(0.0, device=self.device)
                    # If this timestep is fully padded, append zero annotation to keep length
                    if self.critic_replay_scale > 0.0 and not skip_ac_batch:
                        replay_value_annotations.append(
                            torch.zeros_like(sample_mask, device=self.device)
                        )

                    # Decode dreamed states for visualization (every 50 steps)
                    if self.train_step % 50 == 0 and self.use_pixels:
                        with torch.no_grad():
                            try:
                                # dreamed_recurrent_states: (n_dream_steps, batch, d_hidden)
                                n_steps, batch_sz = dreamed_recurrent_states.shape[:2]
                                # Flatten to (n_steps * batch, features) for decoder
                                states_flat = dreamed_recurrent_states.view(
                                    n_steps * batch_sz, -1
                                )
                                dreamed_obs = self.world_model.decoder(states_flat)
                                pixels_flat = dreamed_obs.get("pixels")
                                if pixels_flat is not None:
                                    # Decoder flattens to (n_steps * batch, C, H, W), reshape back
                                    C, H, W = pixels_flat.shape[1:]
                                    last_dreamed_pixels = torch.sigmoid(
                                        pixels_flat.view(n_steps, batch_sz, C, H, W)
                                    ).detach()
                            except Exception:
                                last_dreamed_pixels = None

                # Accumulate losses (per-sample masking handled inside loss functions)
                if total_wm_loss is None:
                    total_wm_loss = wm_loss
                    total_actor_loss = actor_loss
                    total_critic_loss = critic_loss
                else:
                    total_wm_loss = total_wm_loss + wm_loss  # type: ignore
                    total_actor_loss = total_actor_loss + actor_loss  # type: ignore
                    total_critic_loss = total_critic_loss + critic_loss  # type: ignore

            # --- Replay critic grounding (uses imagination annotations) ---
            if (
                self.critic_replay_scale > 0.0
                and not skip_ac_batch
                and ac_any
                and len(replay_value_annotations) == effective_train_steps
                and len(replay_posterior_states) == effective_train_steps
            ):
                # Stack replay data
                replay_posterior = torch.stack(
                    replay_posterior_states, dim=1
                )  # [B, T, D]
                replay_annotations = torch.stack(
                    replay_value_annotations, dim=0
                )  # [T, B]

                replay_rewards = self.rewards[:, train_start_t:].transpose(
                    0, 1
                )  # [T', B]
                replay_continues = (1.0 - self.is_terminal.float()).transpose(0, 1)[
                    train_start_t:
                ]  # [T', B]
                replay_mask = self.mask[:, train_start_t:].transpose(0, 1)  # [T', B]

                # Compute replay lambda-returns with annotations (continues are probabilities)
                replay_lambda_returns = calculate_lambda_returns(
                    replay_rewards,
                    replay_rewards,
                    replay_continues,
                    self.gamma,
                    self.lam,
                    effective_train_steps,
                    value_annotations=replay_annotations,
                    continues_are_logits=False,
                ).transpose(0, 1)  # [B, T]

                # Critic logits on posterior states (gradients to critic only)
                replay_logits = self.critic(
                    replay_posterior.detach()
                )  # [B, T, num_bins]
                logits_flat = replay_logits.reshape(-1, replay_logits.size(-1))
                targets_flat = replay_lambda_returns.detach().reshape(-1)
                mask_flat = replay_mask.reshape(-1).float()

                targets_twohot = twohot_encode(targets_flat, self.B)
                per_step_ce = -torch.sum(
                    targets_twohot * F.log_softmax(logits_flat, dim=-1), dim=-1
                )
                replay_loss = (per_step_ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)

                # EMA regularizer for replay pass (distributional)
                with torch.no_grad():
                    replay_logits_ema = self.critic_ema(replay_posterior.detach())
                ema_logits_flat = replay_logits_ema.reshape(
                    -1, replay_logits_ema.size(-1)
                )
                ema_probs = F.softmax(ema_logits_flat, dim=-1)
                per_step_ema = -torch.sum(
                    ema_probs * F.log_softmax(logits_flat, dim=-1), dim=-1
                )
                replay_ema_reg = (per_step_ema * mask_flat).sum() / (
                    mask_flat.sum() + 1e-8
                )

                replay_loss_total = (
                    replay_loss + self.critic_ema_regularizer * replay_ema_reg
                )
                total_critic_loss = (
                    total_critic_loss + self.critic_replay_scale * replay_loss_total
                )  # type: ignore

            # End forward, start backward

            # Compute surprise BEFORE backward pass (proactive, not reactive)
            seq_len = self.states.shape[1] if hasattr(self, "states") else 1
            self.compute_surprise_for_batch(total_wm_loss, seq_len)

            # Backprop
            assert (
                total_wm_loss is not None
                and total_actor_loss is not None
                and total_critic_loss is not None
            )
            if not torch.isfinite(total_wm_loss):
                raise RuntimeError(
                    f"Non-finite WM loss at step {self.train_step}: {total_wm_loss.item()}"
                )

            # Use the batch-level AC skip decision (or skip if no valid AC steps)
            if skip_ac_batch or not ac_any:
                # WM focus mode: only update WM this step
                total_wm_loss.backward()
                adaptive_gradient_clipping(self.wm_params)
                self.wm_optimizer.step()
            else:
                # Full training: WM + critic + actor
                # Apply surprise-based lr scaling to AC optimizers (uses smoothed surprise)
                lr_scale = self.get_ac_lr_scale()
                if lr_scale < 1.0:
                    for pg in self.actor_optimizer.param_groups:
                        pg["lr"] = self.base_actor_lr * lr_scale
                    for pg in self.critic_optimizer.param_groups:
                        pg["lr"] = self.base_critic_lr * lr_scale

                total_wm_loss.backward()
                total_critic_loss.backward()
                total_actor_loss.backward()
                # AGC: clip gradients based on param/grad norm ratio (DreamerV3)
                # Use more aggressive clipping for pixel observations (prevent NaN)
                agc_clip = 0.15 if self.use_pixels else 0.3
                adaptive_gradient_clipping(self.wm_params, clip_factor=agc_clip)
                adaptive_gradient_clipping(
                    self.critic.parameters(), clip_factor=agc_clip
                )
                adaptive_gradient_clipping(
                    self.actor.parameters(), clip_factor=agc_clip
                )
                self.wm_optimizer.step()
                self.critic_optimizer.step()
                self.actor_optimizer.step()

                # Polyak update for critic EMA
                with torch.no_grad():
                    for param, param_ema in zip(
                        self.critic.parameters(), self.critic_ema.parameters()
                    ):
                        param_ema.data.mul_(self.critic_ema_decay).add_(
                            param.data, alpha=1 - self.critic_ema_decay
                        )

                    # Reset lr after step (for next iteration's scaling)
                    if lr_scale < 1.0:
                        for pg in self.actor_optimizer.param_groups:
                            pg["lr"] = self.base_actor_lr
                        for pg in self.critic_optimizer.param_groups:
                            pg["lr"] = self.base_critic_lr

            # End backward

            log_step = self.train_step
            self.train_step += 1

            # Log metrics to MLflow
            sequence_length = effective_train_steps if hasattr(self, "states") else 0
            self.log_metrics(
                total_wm_loss,
                total_actor_loss,
                total_critic_loss,
                wm_loss_components,
                sequence_length,
                dreamed_rewards_list,
                dreamed_values_list,
                actor_entropy_list,
                replay_loss,
                replay_ema_reg,
                last_obs_pixels,
                last_obs_pixels_original,
                last_reconstruction_pixels,
                last_posterior_probs,
                last_dreamed_pixels,
                log_step,
            )

            # Send models to collector periodically
            if self.train_step % self.steps_per_weight_sync == 0:
                self.send_models_to_collector(self.train_step)

            # Periodic logging (every 100 steps)
            self.steps_since_log += 1
            if self.train_step % 100 == 0:
                elapsed = time.time() - self.last_log_time
                steps_per_sec = self.steps_since_log / elapsed if elapsed > 0 else 0
                seq_len = sequence_length if sequence_length > 0 else 1
                eta_hours = (
                    (self.max_train_steps - self.train_step) / steps_per_sec / 3600
                    if steps_per_sec > 0
                    else 0
                )

                print(
                    f"Step {self.train_step}/{self.max_train_steps} | "
                    f"{steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h | "
                    f"WM: {total_wm_loss.item() / seq_len:.4f} | "
                    f"Actor: {total_actor_loss.item() / seq_len:.4f} | "
                    f"Critic: {total_critic_loss.item() / seq_len:.4f}"
                )

                self.logger.add_scalar(
                    "train/throughput", steps_per_sec, self.train_step
                )
                # Universal progress axes (environment and wall-clock)
                self.logger.add_scalar(
                    "train/updates_total", float(self.train_step), self.train_step
                )
                self.logger.add_scalar(
                    "env/frames_total",
                    float(
                        self.replay_buffer.total_env_steps
                        + self._resume_env_steps_offset
                    ),
                    self.train_step,
                )
                env_frames_total = float(
                    self.replay_buffer.total_env_steps + self._resume_env_steps_offset
                )
                self.logger.add_scalar(
                    "train/updates_per_env_step",
                    float(self.train_step) / max(1.0, env_frames_total),
                    self.train_step,
                )
                self.logger.add_scalar(
                    "env/episodes_total",
                    float(self.replay_buffer.total_episodes_added),
                    self.train_step,
                )
                self.logger.add_scalar(
                    "time/elapsed_sec",
                    time.time() - self.train_start_time,
                    self.train_step,
                )

                # Log avg episode length for convergence context (not solved metric)
                avg_ep_len = self.replay_buffer.recent_avg_episode_length
                if avg_ep_len > 0:
                    self.logger.add_scalar(
                        "env/episode_length", avg_ep_len, self.train_step
                    )
                self.last_log_time = time.time()
                self.steps_since_log = 0

            if (
                self.eval_every > 0
                and self.eval_episodes > 0
                and self.train_step > 0
                and self.train_step % self.eval_every == 0
            ):
                eval_avg_len = self.evaluate_policy(
                    self.eval_episodes, step=self.train_step
                )
                # Early stopping based on eval (deterministic policy)
                if (
                    self.early_stop_ep_length > 0
                    and eval_avg_len >= self.early_stop_ep_length
                ):
                    print(
                        f"SOLVED! eval_avg_len={eval_avg_len:.1f} >= {self.early_stop_ep_length}"
                    )
                    break

            # Checkpoint every N steps (skip in dry_run)
            if (
                not self.dry_run
                and self.train_step > 0
                and self.train_step % self.checkpoint_interval == 0
            ):
                self.save_checkpoint()

        # Final save (skip in dry_run)
        if not self.dry_run:
            self.save_checkpoint(final=True)
            print(f"Training complete. Final checkpoint saved to {self.checkpoint_dir}")
        else:
            print("DRY RUN complete - no checkpoint saved.")

        # Cleanup
        self.replay_buffer.stop()
        self.logger.close()

    def update_return_scale(self, lambda_returns, decay=0.99):
        flat = lambda_returns.detach().reshape(-1)
        range_batch = (
            torch.quantile(flat, 0.95).item() - torch.quantile(flat, 0.05).item()
        )
        self.S = self.S * decay + range_batch * (1 - decay)

    def _surprise_log_ratio(self, key: str, value: float) -> float | None:
        """Return log(value / EMA) and update EMA; None if EMA not initialized."""
        ema = self._wm_surprise_ema.get(key)
        if ema is None:
            self._wm_surprise_ema[key] = value
            return None
        log_ratio = math.log(
            (value + self._wm_surprise_eps) / (ema + self._wm_surprise_eps)
        )
        self._wm_surprise_ema[key] = (
            self.surprise_ema_beta * ema + (1.0 - self.surprise_ema_beta) * value
        )
        return log_ratio

    def get_cosine_schedule(self, max_val: float, min_val: float) -> float:
        """Cosine schedule from max_val to min_val over training."""
        progress = min(1.0, self.train_step / max(1, self.max_train_steps))
        return min_val + 0.5 * (max_val - min_val) * (1 + math.cos(math.pi * progress))

    def get_current_wm_ac_ratio(self) -> int:
        """Get current WM:AC ratio (possibly scheduled)."""
        if not self.wm_ac_ratio_cosine:
            return self.wm_ac_ratio
        # Normal: max→min (8→2), Inverted: min→max (2→8)
        if self.wm_ac_ratio_invert:
            ratio = self.get_cosine_schedule(
                float(self.wm_ac_ratio_min), float(self.wm_ac_ratio_max)
            )
        else:
            ratio = self.get_cosine_schedule(
                float(self.wm_ac_ratio_max), float(self.wm_ac_ratio_min)
            )
        return max(1, round(ratio))

    def apply_lr_schedule(self):
        """Apply cosine LR decay to all optimizers."""
        if not self.lr_cosine_decay:
            return
        scale = self.get_cosine_schedule(1.0, self.lr_cosine_min_factor)
        for pg in self.wm_optimizer.param_groups:
            pg["lr"] = self.config.wm_lr * scale
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = self.base_actor_lr * scale
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.base_critic_lr * scale

    def compute_surprise_for_batch(self, wm_loss: torch.Tensor, seq_len: int) -> float:
        """
        Compute surprise from current batch's WM loss BEFORE backward pass.
        Updates EMA, smoothed surprise, and WM focus mode. Call this right after forward pass.

        Returns the raw surprise log ratio.
        """
        wm_per_step = wm_loss.item() / max(1, seq_len)

        # Compute raw surprise (log ratio vs EMA)
        raw_surprise = self._surprise_log_ratio("total", wm_per_step)
        if raw_surprise is None:
            raw_surprise = 0.0

        self._last_surprise_log_ratio = raw_surprise

        # Update smoothed surprise (lingers after spikes)
        # Only positive surprise affects smoothing (WM struggling)
        positive_surprise = max(0.0, raw_surprise)
        # Smoothed surprise takes the max of decay and new value (ratchet up quickly, decay slowly)
        decayed = self._smoothed_surprise * self.surprise_smooth_beta
        self._smoothed_surprise = max(decayed, positive_surprise)

        # Trigger WM focus mode if surprise exceeds threshold (respecting cooldown)
        if positive_surprise > self.surprise_wm_focus_threshold:
            if self._wm_focus_cooldown_remaining <= 0:
                # Not in cooldown, can enter focus
                self._wm_focus_steps_remaining = self.surprise_wm_focus_duration
                self._wm_focus_ac_counter = 0  # Reset counter
            # If in cooldown, LR scaling still applies but no focus mode

        # Tick cooldown
        if self._wm_focus_cooldown_remaining > 0:
            self._wm_focus_cooldown_remaining -= 1

        return raw_surprise

    def evaluate_policy(self, num_episodes: int, step: int) -> float:
        """Run deterministic evaluation episodes and log summary metrics.

        Returns:
            Average episode length across evaluation episodes.
        """
        if num_episodes <= 0:
            return 0.0

        was_training = (
            self.encoder.training,
            self.world_model.training,
            self.actor.training,
        )
        self.encoder.eval()
        self.world_model.eval()
        self.actor.eval()

        h_prev_backup = self.world_model.h_prev.clone()

        env = create_env(
            self.config.environment_name,
            use_pixels=self.use_pixels,
            config=self.config,
        )
        target_size = self.config.encoder_cnn_target_size if self.use_pixels else None

        episode_lengths = []
        episode_rewards = []

        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _info = env.reset()
                h = torch.zeros(
                    1,
                    self.config.d_hidden * self.config.rnn_n_blocks,
                    device=self.device,
                )
                action_onehot = torch.zeros(1, self.n_actions, device=self.device)
                z_prev = torch.zeros(
                    1, self.num_latents, self.num_classes, device=self.device
                )
                z_prev_embed = self.world_model.z_embedding(z_prev.view(1, -1))

                total_reward = 0.0
                steps = 0

                while True:
                    if self.use_pixels:
                        pixel_obs_t = (
                            torch.from_numpy(obs["pixels"])
                            .to(self.device)
                            .float()
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                        )
                        if target_size and pixel_obs_t.shape[-2:] != tuple(target_size):
                            pixel_obs_t = resize_pixels_to_target(
                                pixel_obs_t, target_size
                            )
                        vec_obs_t = (
                            torch.from_numpy(obs["state"])
                            .to(self.device)
                            .float()
                            .unsqueeze(0)
                        )
                        vec_obs_t = symlog(vec_obs_t)
                        encoder_input = {"pixels": pixel_obs_t, "state": vec_obs_t}
                    else:
                        vec_obs_t = (
                            torch.from_numpy(obs).to(self.device).float().unsqueeze(0)
                        )
                        vec_obs_t = symlog(vec_obs_t)
                        encoder_input = vec_obs_t

                    h, _ = self.world_model.step_dynamics(
                        z_prev_embed, action_onehot, h
                    )

                    posterior_logits = self.encoder(encoder_input)
                    posterior_logits_mixed = unimix_logits(
                        posterior_logits, unimix_ratio=0.01
                    )
                    z_indices = posterior_logits_mixed.argmax(dim=-1)
                    z_onehot = F.one_hot(
                        z_indices, num_classes=self.d_hidden // 16
                    ).float()
                    z_sample = z_onehot

                    actor_input = self.world_model.join_h_and_z(h, z_sample)
                    action_logits = self.actor(actor_input)
                    action = action_logits.argmax(dim=-1)
                    action_onehot = F.one_hot(
                        action, num_classes=self.n_actions
                    ).float()
                    z_prev_embed = self.world_model.z_embedding(z_sample.view(1, -1))

                    obs, reward, terminated, truncated, _info = env.step(action.item())
                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break

                episode_lengths.append(steps)
                episode_rewards.append(total_reward)

        env.close()
        self.world_model.h_prev = h_prev_backup

        if was_training[0]:
            self.encoder.train()
        if was_training[1]:
            self.world_model.train()
        if was_training[2]:
            self.actor.train()

        avg_len = sum(episode_lengths) / len(episode_lengths)
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        win_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)

        self.logger.add_scalar("eval/episode_length", avg_len, step)
        self.logger.add_scalar("eval/episode_reward", avg_reward, step)
        self.logger.add_scalar("eval/win_rate", win_rate, step)

        # Duplicate eval metrics with env-frame step for apples-to-apples sample-efficiency plots.
        env_frames_total = int(
            self.replay_buffer.total_env_steps + self._resume_env_steps_offset
        )
        self.logger.add_scalar("eval_frames/episode_length", avg_len, env_frames_total)
        self.logger.add_scalar(
            "eval_frames/episode_reward", avg_reward, env_frames_total
        )
        self.logger.add_scalar("eval_frames/win_rate", win_rate, env_frames_total)

        return avg_len

    def should_skip_ac_for_wm_focus(self) -> bool:
        """
        Check if AC update should be skipped this step due to WM focus mode.
        In focus mode, only update AC every N steps (ratio).
        """
        if self._wm_focus_steps_remaining <= 0:
            return False

        # Decrement focus countdown
        self._wm_focus_steps_remaining -= 1

        # Start cooldown when exiting focus mode
        if self._wm_focus_steps_remaining == 0:
            self._wm_focus_cooldown_remaining = self.surprise_wm_focus_cooldown

        # Check if this is an AC step (every Nth step)
        self._wm_focus_ac_counter += 1
        if self._wm_focus_ac_counter >= self.surprise_wm_focus_ratio:
            self._wm_focus_ac_counter = 0
            return False  # Do AC this step

        return True  # Skip AC this step

    def get_ac_lr_scale(self) -> float:
        """
        Get AC learning rate scale based on smoothed surprise.

        Returns scale factor in (0, 1]:
        - High surprise → low scale (slower AC learning)
        - Low/negative surprise → scale ≈ 1 (normal learning)

        Uses smoothed surprise so LR reduction lingers after spikes.
        """
        if not self.surprise_scale_ac_lr:
            return 1.0
        # Use smoothed surprise (not raw) so reduction lingers
        return 1.0 / (1.0 + self._smoothed_surprise * self.surprise_lr_scale_k)

    def _get_model(self, model):
        """Get underlying model (handles both compiled and non-compiled)."""
        return getattr(model, "_orig_mod", model)

    def save_checkpoint(self, final=False):
        """Save all model checkpoints."""
        suffix = "final" if final else f"step_{self.train_step}"
        checkpoint = {
            "step": self.train_step,
            "encoder": self._get_model(self.encoder).state_dict(),
            "world_model": {
                k: v
                for k, v in self._get_model(self.world_model).state_dict().items()
                if k not in ("h_prev", "z_prev")
            },
            "actor": self._get_model(self.actor).state_dict(),
            "critic": self._get_model(self.critic).state_dict(),
            "critic_ema": self._get_model(self.critic_ema).state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "return_scale": self.S,
            "surprise_ema": self._wm_surprise_ema,
            "smoothed_surprise": self._smoothed_surprise,
            "mlflow_run_id": self.mlflow_run_id,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load full checkpoint (encoder, world model, actor, critic, optimizers).

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load encoder and world_model
        self._get_model(self.encoder).load_state_dict(checkpoint["encoder"])
        self._get_model(self.world_model).load_state_dict(
            checkpoint["world_model"], strict=False
        )

        # Load actor/critic
        if "actor" in checkpoint:
            self._get_model(self.actor).load_state_dict(checkpoint["actor"])
            self._get_model(self.critic).load_state_dict(checkpoint["critic"])
            if "critic_ema" in checkpoint:
                self._get_model(self.critic_ema).load_state_dict(
                    checkpoint["critic_ema"]
                )
            else:
                # Backward compatibility: old checkpoints did not save critic_ema.
                self._get_model(self.critic_ema).load_state_dict(
                    self._get_model(self.critic).state_dict()
                )

        # Restore optimizers (skip if architecture changed)
        if "wm_optimizer" in checkpoint:
            try:
                self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
            except ValueError as e:
                if "doesn't match the size" in str(e):
                    print(
                        f"Warning: Skipping WM optimizer state (architecture changed)"
                    )
                else:
                    raise
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        # Restore auxiliary training state when available.
        self.S = float(checkpoint.get("return_scale", self.S))
        surprise_ema = checkpoint.get("surprise_ema")
        if isinstance(surprise_ema, dict):
            self._wm_surprise_ema = {str(k): float(v) for k, v in surprise_ema.items()}
        self._smoothed_surprise = float(
            checkpoint.get("smoothed_surprise", self._smoothed_surprise)
        )

        self.train_step = checkpoint.get("step", 0)
        print(f"Resumed checkpoint from {checkpoint_path} at step {self.train_step}")

    def log_metrics(
        self,
        total_wm_loss,
        total_actor_loss,
        total_critic_loss,
        wm_loss_components,
        sequence_length,
        dreamed_rewards_list,
        dreamed_values_list,
        actor_entropy_list,
        replay_loss=None,
        replay_ema_reg=None,
        last_obs_pixels=None,
        last_obs_pixels_original=None,
        last_reconstruction_pixels=None,
        last_posterior_probs=None,
        last_dreamed_pixels=None,
        step=None,
    ):
        """Log metrics to MLflow.

        Metric naming convention:
        - loss/{model}/total: Top-level loss per model (wm, actor, critic)
        - wm/{component}/loss: World model sub-component losses
        - wm/rssm/*: RSSM-specific metrics (KL divergences)
        - dream/{source}/*: Imagined trajectory stats from world model rollouts
        - actor/*: Actor network metrics
        - train/*: Training infrastructure metrics
        - env/*: Environment interaction metrics
        - viz/*: Visualization artifacts
        """
        if step is None:
            step = self.train_step
        log_scalars = step % self.log_every == 0
        log_images = step % self.image_log_every == 0
        if not (log_scalars or log_images):
            return

        # Safety check - should not happen due to assertions, but just in case
        if (
            total_wm_loss is None
            or total_actor_loss is None
            or total_critic_loss is None
        ):
            return

        if log_scalars:
            is_full_profile = self.log_profile == "full"
            # All losses normalized to per-step for fair comparison
            if sequence_length > 0:
                norm = 1.0 / sequence_length
                beta_pred = self.config.beta_pred
                beta_dyn = self.config.beta_dyn
                beta_rep = self.config.beta_rep

                # Convert tensor loss components to CPU floats (single sync for all 8 components)
                wm_components_cpu = {k: v.item() for k, v in wm_loss_components.items()}

                # Per-step totals for each model
                wm_per_step = total_wm_loss.item() * norm
                actor_per_step = total_actor_loss.item() * norm
                critic_per_step = total_critic_loss.item() * norm
                self.logger.add_scalar("loss/wm/total", wm_per_step, step)
                self.logger.add_scalar("loss/actor/total", actor_per_step, step)
                self.logger.add_scalar("loss/critic/total", critic_per_step, step)

                # Replay critic grounding (logged as raw masked average loss)
                if replay_loss is not None:
                    replay_loss_value = float(replay_loss.item())
                    self.logger.add_scalar(
                        "loss/critic/replay", replay_loss_value, step
                    )
                if replay_ema_reg is not None:
                    self.logger.add_scalar(
                        "loss/critic/replay_ema_reg", float(replay_ema_reg.item()), step
                    )

                # Raw component values (per-step, unscaled)
                pixel = wm_components_cpu["prediction_pixel"] * norm
                state = wm_components_cpu["prediction_vector"] * norm
                reward = wm_components_cpu["prediction_reward"] * norm
                cont = wm_components_cpu["prediction_continue"] * norm
                dyn = wm_components_cpu["dynamics"] * norm
                rep = wm_components_cpu["representation"] * norm

                # Surprise signals: log-ratio of recent loss vs EMA
                # Note: "total" is computed in compute_surprise_for_batch (before backward)
                # Only compute component surprises here to avoid double EMA update
                surprise_inputs = {
                    "pixel": pixel,
                    "state": state,
                    "reward": reward,
                    "continue": cont,
                    "kl_dyn": dyn,
                    "kl_rep": rep,
                }
                surprise_log_ratios = {}
                for key, value in surprise_inputs.items():
                    log_ratio = self._surprise_log_ratio(key, float(value))
                    if log_ratio is not None:
                        surprise_log_ratios[key] = log_ratio

                # Log total surprise (already computed in compute_surprise_for_batch)
                if is_full_profile:
                    self.logger.add_scalar("wm/surprise/ready", 1.0, step)
                    self.logger.add_scalar(
                        "wm/surprise/total_log_ratio",
                        self._last_surprise_log_ratio,
                        step,
                    )

                # Log component surprises
                if is_full_profile and "reward" in surprise_log_ratios:
                    self.logger.add_scalar(
                        "wm/surprise/reward_log_ratio",
                        surprise_log_ratios["reward"],
                        step,
                    )
                if is_full_profile and "continue" in surprise_log_ratios:
                    self.logger.add_scalar(
                        "wm/surprise/continue_log_ratio",
                        surprise_log_ratios["continue"],
                        step,
                    )
                if is_full_profile and "pixel" in surprise_log_ratios:
                    self.logger.add_scalar(
                        "wm/surprise/pixel_log_ratio",
                        surprise_log_ratios["pixel"],
                        step,
                    )
                if is_full_profile and "state" in surprise_log_ratios:
                    self.logger.add_scalar(
                        "wm/surprise/state_log_ratio",
                        surprise_log_ratios["state"],
                        step,
                    )
                if is_full_profile and "kl_dyn" in surprise_log_ratios:
                    self.logger.add_scalar(
                        "wm/surprise/kl_dyn_log_ratio",
                        surprise_log_ratios["kl_dyn"],
                        step,
                    )
                if is_full_profile and "kl_rep" in surprise_log_ratios:
                    self.logger.add_scalar(
                        "wm/surprise/kl_rep_log_ratio",
                        surprise_log_ratios["kl_rep"],
                        step,
                    )
                component_vals = list(surprise_log_ratios.values())
                if is_full_profile and component_vals:
                    self.logger.add_scalar(
                        "wm/surprise/max_component_log_ratio",
                        max(component_vals),
                        step,
                    )

                # Log AC learning rate scale (based on surprise)
                if is_full_profile and self.surprise_scale_ac_lr:
                    lr_scale = self.get_ac_lr_scale()
                    self.logger.add_scalar("ac/lr_scale", lr_scale, step)
                    # Log smoothed vs raw surprise for debugging
                    self.logger.add_scalar(
                        "wm/surprise/smoothed", self._smoothed_surprise, step
                    )
                    # Log WM focus mode (1 = in focus, 0 = normal) and cooldown
                    self.logger.add_scalar(
                        "ac/wm_focus_active",
                        float(self._wm_focus_steps_remaining > 0),
                        step,
                    )
                    self.logger.add_scalar(
                        "ac/wm_focus_cooldown",
                        float(self._wm_focus_cooldown_remaining > 0),
                        step,
                    )

                # World model sub-component losses (grouped by module)
                # Decoder losses (reconstruction)
                if self._has_pixel_obs:
                    self.logger.add_scalar("wm/decoder/pixel_loss", pixel, step)
                if self._has_vector_obs:
                    self.logger.add_scalar("wm/decoder/state_loss", state, step)
                # Predictor head losses
                self.logger.add_scalar("wm/reward_head/loss", reward, step)
                self.logger.add_scalar("wm/continue_head/loss", cont, step)
                # RSSM KL losses (after free bits)
                self.logger.add_scalar("wm/rssm/kl_dynamics", dyn, step)
                self.logger.add_scalar("wm/rssm/kl_representation", rep, step)

                # Scaled contributions to total (these should sum to loss/wm/total)
                pred_total = pixel + state + reward + cont
                if is_full_profile:
                    self.logger.add_scalar(
                        "wm/scaled/prediction", beta_pred * pred_total, step
                    )
                    self.logger.add_scalar("wm/scaled/dynamics", beta_dyn * dyn, step)
                    self.logger.add_scalar(
                        "wm/scaled/representation", beta_rep * rep, step
                    )

                # Raw KL divergences (before free bits clipping, for debugging)
                kl_dyn_raw = wm_components_cpu["kl_dynamics_raw"] * norm
                kl_rep_raw = wm_components_cpu["kl_representation_raw"] * norm
                self.logger.add_scalar("wm/rssm/kl_dynamics_raw", kl_dyn_raw, step)
                self.logger.add_scalar(
                    "wm/rssm/kl_representation_raw", kl_rep_raw, step
                )
            else:
                # Fallback if no sequence
                self.logger.add_scalar("loss/wm/total", total_wm_loss.item(), step)
                self.logger.add_scalar(
                    "loss/actor/total", total_actor_loss.item(), step
                )
                self.logger.add_scalar(
                    "loss/critic/total", total_critic_loss.item(), step
                )

            # Dreamed trajectory statistics (from world model imagination rollouts)
            # Batch stats computation and sync once to reduce GPU stalls
            if dreamed_rewards_list:
                all_dreamed_rewards = torch.cat(dreamed_rewards_list, dim=0)
                reward_stats = torch.stack(
                    [
                        all_dreamed_rewards.mean(),
                        all_dreamed_rewards.std(),
                        all_dreamed_rewards.min(),
                        all_dreamed_rewards.max(),
                    ]
                ).cpu()  # Single sync for all 4 stats
                # Rewards predicted by WM reward head during imagination
                self.logger.add_scalar(
                    "dream/wm_reward/mean", reward_stats[0].item(), step
                )
                self.logger.add_scalar(
                    "dream/wm_reward/std", reward_stats[1].item(), step
                )
                if is_full_profile:
                    self.logger.add_scalar(
                        "dream/wm_reward/min", reward_stats[2].item(), step
                    )
                    self.logger.add_scalar(
                        "dream/wm_reward/max", reward_stats[3].item(), step
                    )

            if dreamed_values_list:
                all_dreamed_values = torch.cat(dreamed_values_list, dim=0)
                value_stats = torch.stack(
                    [
                        all_dreamed_values.mean(),
                        all_dreamed_values.std(),
                    ]
                ).cpu()  # Single sync for both stats
                # Values from critic during imagination
                self.logger.add_scalar(
                    "dream/critic_value/mean", value_stats[0].item(), step
                )
                self.logger.add_scalar(
                    "dream/critic_value/std", value_stats[1].item(), step
                )
                symlog_values = symlog(all_dreamed_values)
                symlog_stats = torch.stack(
                    [
                        symlog_values.mean(),
                        symlog_values.std(),
                    ]
                ).cpu()
                self.logger.add_scalar(
                    "dream/critic_value_symlog/mean", symlog_stats[0].item(), step
                )
                self.logger.add_scalar(
                    "dream/critic_value_symlog/std", symlog_stats[1].item(), step
                )

            # Actor entropy (important for monitoring exploration)
            if actor_entropy_list:
                all_entropy = torch.stack(actor_entropy_list)
                entropy_stats = torch.stack(
                    [
                        all_entropy.mean(),
                        all_entropy.std(),
                    ]
                ).cpu()  # Single sync for both stats
                self.logger.add_scalar(
                    "actor/entropy/mean", entropy_stats[0].item(), step
                )
                if is_full_profile:
                    self.logger.add_scalar(
                        "actor/entropy/std", entropy_stats[1].item(), step
                    )

            # Learning rates
            if is_full_profile:
                self.logger.add_scalar(
                    "train/lr/wm",
                    self.wm_optimizer.param_groups[0]["lr"],
                    step,
                )
                self.logger.add_scalar(
                    "train/lr/actor",
                    self.actor_optimizer.param_groups[0]["lr"],
                    step,
                )
                self.logger.add_scalar(
                    "train/lr/critic",
                    self.critic_optimizer.param_groups[0]["lr"],
                    step,
                )

            # Log WM:AC ratio (if using cosine schedule)
            if is_full_profile and self.wm_ac_ratio_cosine:
                self.logger.add_scalar(
                    "train/wm_ac_ratio",
                    self.get_current_wm_ac_ratio(),
                    step,
                )

        # Visualizations every 250 steps (show first sample only)
        if (
            # step % 50 == 0
            log_images
            and last_obs_pixels is not None
            and last_reconstruction_pixels is not None
        ):
            # Take first sample from batch
            actual = (last_obs_pixels[0] / 255.0).clamp(0, 1)  # (C, H, W)
            recon = torch.sigmoid(last_reconstruction_pixels[0]).clamp(
                0, 1
            )  # (C, H, W)
            # Resize reconstruction to match actual if needed
            if recon.shape != actual.shape:
                recon = F.interpolate(
                    recon.unsqueeze(0),
                    size=actual.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # 1. Actual vs Reconstruction side by side (decoder output quality)
            comparison = torch.cat([actual, recon], dim=2)  # concat on width
            self.logger.add_image("viz/decoder/reconstruction", comparison, step)

            # 2. Reconstruction error heatmap (decoder error visualization)
            error = torch.abs(actual - recon).mean(dim=0, keepdim=True)  # (1, H, W)
            error_norm = error / (error.max() + 1e-8)
            error_heatmap = error_norm.repeat(3, 1, 1)  # grayscale to RGB
            self.logger.add_image("viz/decoder/error", error_heatmap, step)

            # 3. Latent activation heatmap (encoder posterior distribution)
            if last_posterior_probs is not None:
                # Shape: (batch, d_hidden, categories) -> take first batch, make 2D
                latent_probs = last_posterior_probs[0]  # (512, 32)
                # Normalize and add batch/channel dims for add_images
                latent_img = latent_probs.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 32)
                self.logger.add_images("viz/encoder/latent_posterior", latent_img, step)

            # 4. Dream rollout video (WM imagination/planning visualization)
            if last_dreamed_pixels is not None:
                # Shape: (n_dream_steps, batch, C, H, W) -> take first batch
                dream_frames = last_dreamed_pixels[:, 0]  # (n_steps, C, H, W)
                # Resize to match actual size if needed
                if dream_frames.shape[2:] != actual.shape[1:]:
                    dream_frames = F.interpolate(
                        dream_frames,
                        size=actual.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                # Add video: shape (N, T, C, H, W) - batch, time, channels, height, width
                video = dream_frames.unsqueeze(0)  # (1, n_steps, C, H, W)
                self.logger.add_video("viz/wm/dream_video", video, step, fps=4)
                # Also add strip image for quick glance
                n_show = min(5, dream_frames.shape[0])
                dream_strip = torch.cat([dream_frames[i] for i in range(n_show)], dim=2)
                self.logger.add_images(
                    "viz/wm/dream_strip", dream_strip.unsqueeze(0), step
                )

            # 5. Original resolution image (environment frame)
            if last_obs_pixels_original is not None:
                original = (last_obs_pixels_original[0] / 255.0).clamp(
                    0, 1
                )  # First sample only
                self.logger.add_image("viz/env/frame", original, step)

        self.logger.flush()

    def send_models_to_collector(self, training_step):
        # Exclude h_prev/z_prev buffers as they have batch-size-dependent shapes
        wm = self._get_model(self.world_model)
        actor = self._get_model(self.actor)
        encoder = self._get_model(self.encoder)

        wm_state = {
            k: v.cpu()
            for k, v in wm.state_dict().items()
            if k not in ("h_prev", "z_prev")
        }
        models_to_send = {
            "actor": {k: v.cpu() for k, v in actor.state_dict().items()},
            "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
            "world_model": wm_state,
        }
        try:
            # Clear queue to ensure collector gets the latest version
            cleared = 0
            while not self.model_queue.empty():
                self.model_queue.get_nowait()
                cleared += 1
            self.model_queue.put_nowait(models_to_send)
            print(
                f"Trainer: Sent models at step {training_step} (cleared {cleared} old)"
            )
        except Full:
            print("Trainer: Model queue was full. Skipping update.")
        except Exception as e:
            print(f"Trainer: Failed to send models: {type(e).__name__}: {e}")


def train_world_model(
    config,
    data_queue,
    model_queue,
    log_dir,
    checkpoint_path=None,
    mlflow_run_id=None,
    dry_run=False,
):
    # Set MLflow tracking URI and join existing run in this child process (skip in dry_run)
    if mlflow_run_id and not dry_run:
        mlruns_dir = os.path.join(os.path.dirname(log_dir), "mlruns")
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
        # Join the existing run (subprocess doesn't inherit active run from parent)
        mlflow.start_run(run_id=mlflow_run_id)

    trainer = WorldModelTrainer(
        config,
        data_queue,
        model_queue,
        log_dir,
        checkpoint_path,
        mlflow_run_id,
        dry_run,
    )
    trainer.train_models()
