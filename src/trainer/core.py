import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from queue import Full
import mlflow
import os
import time

from ..replay_buffer import EpisodeReplayBuffer
from .math_utils import symlog, symexp, twohot_encode, resize_pixels_to_target, unimix_logits
from .model_init import initialize_actor, initialize_critic, initialize_world_model
from .profiling import ProfilerManager, TimingAccumulator
from .losses import compute_actor_critic_losses
from .mlflow_logger import MLflowLogger


class WorldModelTrainer:
    def __init__(
        self,
        config,
        data_queue,
        model_queue,
        log_dir,
        checkpoint_path=None,
        mode="train",
        reset_ac=False,
        mlflow_run_id=None,
    ):
        self.config = config  # Store config for use in methods
        self.device = torch.device(config.general.device)
        self.use_pixels = config.general.use_pixels
        self.n_dream_steps = config.train.num_dream_steps
        self.gamma = config.train.gamma
        self.lam = config.train.lam
        self.actor_entropy_coef = config.train.actor_entropy_coef
        self.actor_warmup_steps = config.train.actor_warmup_steps
        b_start = config.train.b_start
        b_end = config.train.b_end
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
            self.device, config, batch_size=config.train.batch_size
        )

        self.compile_models = getattr(config.general, "compile_models", False)
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
        self.wm_optimizer = optim.Adam(
            self.wm_params,
            lr=config.train.wm_lr,
            weight_decay=config.train.weight_decay,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.train.critic_lr,
            weight_decay=config.train.weight_decay,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.train.actor_lr,
            weight_decay=config.train.weight_decay,
        )

        self.max_train_steps = config.train.max_train_steps
        self.train_step = 0
        self.data_queue = data_queue
        self.model_queue = model_queue
        self.d_hidden = config.models.d_hidden
        self.n_actions = config.environment.n_actions
        self.steps_per_weight_sync = config.train.steps_per_weight_sync
        self.batch_size = config.train.batch_size
        self.profile_enabled = getattr(config.general, "profile", False)

        # Replay buffer: background thread drains queue, sample() returns instantly
        self.replay_buffer = EpisodeReplayBuffer(
            data_queue=data_queue,
            max_episodes=config.train.replay_buffer_size,
            min_episodes=config.train.batch_size
            * 2,  # Wait for 2 batches worth before starting
            sequence_length=config.train.sequence_length,
        )
        self.replay_buffer.start()
        print(f"Replay buffer started (max={config.train.replay_buffer_size} episodes)")

        # Initialize MLflow logger with the provided log directory
        self.mlflow_run_id = mlflow_run_id
        self.logger = MLflowLogger(log_dir=log_dir, run_id=mlflow_run_id)
        self.log_dir = log_dir
        print(f"MLflow logging to: {log_dir}")
        self.log_every = 250
        self.image_log_every = 2500

        # Episode length tracking for convergence monitoring

        # Checkpointing
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_interval = 10000

        # Save config snapshot
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            f.write(config.model_dump_json(indent=2))
        print(f"Config saved to: {config_path}")

        # Timing
        self.step_times = []
        self.last_log_time = time.time()
        self.steps_since_log = 0

        # Mode and checkpoint handling
        self.mode = mode
        self.reset_ac = reset_ac

        if checkpoint_path:
            self.checkpoint_type = self.load_checkpoint(
                checkpoint_path, reset_ac=reset_ac
            )
        else:
            self.checkpoint_type = None

        # Mode-specific settings
        if mode == "bootstrap":
            self.actor_warmup_steps = float("inf")  # Never train AC
            print("Bootstrap mode: WM-only training with random actions")
        elif mode == "dreamer":
            self.actor_warmup_steps = 0  # Immediate AC training
            if reset_ac:
                print("Dreamer mode: Fresh actor/critic, keeping WM weights")
            else:
                print("Dreamer mode: Resuming all weights from checkpoint")

    def get_data_from_buffer(self):
        """Sample batch from replay buffer (non-blocking after initial fill)."""
        # Sample returns list of (pixels, states, actions, rewards, terminated) tuples
        # Each already has fixed sequence_length from buffer's _sample_subsequence
        batch = self.replay_buffer.sample(self.batch_size)

        batch_pixels = []
        batch_pixels_original = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_terminated = []

        for pixels, states, actions, rewards, terminated in batch:
            if self.use_pixels and pixels is not None:
                target_size = self.config.models.encoder.cnn.target_size
                # Convert pixels from (T, H, W, C) to (T, C, H, W)
                pixels_tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                batch_pixels_original.append(pixels_tensor)  # Keep original resolution
                if pixels_tensor.shape[-2:] == target_size:
                    pixels_resized = pixels_tensor
                else:
                    pixels_resized = resize_pixels_to_target(
                        pixels_tensor.float(), target_size
                    )
                batch_pixels.append(pixels_resized)

            batch_states.append(torch.from_numpy(states))
            batch_actions.append(torch.from_numpy(actions))
            batch_rewards.append(torch.from_numpy(rewards))
            batch_terminated.append(torch.from_numpy(terminated))

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
        self.terminated = torch.stack(batch_terminated).to(self.device)  # (B, T)

    def train_models(self):
        # === PROFILING: Setup profiler and timing ===
        profiler = ProfilerManager(
            enabled=self.profile_enabled,
            log_dir=self.log_dir,
            component_name="trainer",
            chunk_steps=200,
        )
        profiler.__enter__()
        timing = TimingAccumulator(print_interval=50)

        while self.train_step < self.max_train_steps:
            # --- PROFILE: Data loading ---
            t0 = time.perf_counter()
            self.get_data_from_buffer()  # Sample from replay buffer (non-blocking)
            torch.cuda.synchronize()
            timing.log_phase("data", time.perf_counter() - t0)

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
                terminated_t = self.terminated[:, t_step]

                # Use pre-computed encoder output with unimix (DreamerV3 Section 4)
                posterior_logits = all_posterior_logits[:, t_step]
                posterior_logits_mixed = unimix_logits(posterior_logits, unimix_ratio=0.01)
                posterior_dist = dist.Categorical(
                    logits=posterior_logits_mixed, validate_args=False
                )

                (
                    obs_reconstruction,
                    reward_dist,
                    continue_logits,
                    h_z_joined,
                    prior_logits,
                ) = self.world_model(posterior_dist, action_t)

                # Updating loss of encoder and world model
                wm_loss, wm_loss_dict = self.update_wm_loss(
                    obs_reconstruction,
                    obs_t,
                    reward_dist,
                    reward_t,
                    terminated_t,
                    continue_logits,
                    posterior_dist,
                    prior_logits,
                )

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

                # --- Dream Sequence for Actor-Critic (skip during warmup) ---
                if self.train_step >= self.actor_warmup_steps:
                    h_prev_backup = self.world_model.h_prev.clone()
                    (
                        dreamed_recurrent_states,
                        dreamed_actions_logits,
                        dreamed_actions_sampled,
                    ) = self.dream_sequence(
                        h_z_joined,
                        self.world_model.z_embedding(
                            posterior_dist.probs.view(actual_batch_size, -1)
                        ),
                        self.n_dream_steps,
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
                    dreamed_values_probs = F.softmax(dreamed_values_logits, dim=-1)
                    dreamed_values = torch.sum(dreamed_values_probs * self.B, dim=-1)
                    dreamed_values_list.append(dreamed_values.detach().cpu())

                    lambda_returns = self.calculate_lambda_returns(
                        dreamed_rewards,
                        dreamed_values,
                        dreamed_continues,
                        self.gamma,
                        self.lam,
                        self.n_dream_steps,
                    )

                    self.update_return_scale(lambda_returns)

                    (
                        actor_loss,
                        critic_loss,
                        entropy,
                    ) = compute_actor_critic_losses(
                        dreamed_values_logits,
                        dreamed_values,
                        lambda_returns,
                        dreamed_actions_logits,
                        dreamed_actions_sampled,
                        self.B,
                        self.S,
                        actor_entropy_coef=self.actor_entropy_coef,
                    )
                    actor_entropy_list.append(entropy.detach().cpu())

                    # Decode dreamed states for visualization (every 50 steps)
                    if self.train_step % 50 == 0:
                        with torch.no_grad():
                            try:
                                # dreamed_recurrent_states: (n_dream_steps, batch, d_hidden)
                                n_steps, batch_sz = dreamed_recurrent_states.shape[:2]
                                # Flatten to (n_steps * batch, features) for decoder
                                states_flat = dreamed_recurrent_states.view(
                                    n_steps * batch_sz, -1
                                )
                                dreamed_obs = self.world_model.decoder(states_flat)
                                # Decoder flattens to (n_steps * batch, C, H, W), reshape back
                                pixels_flat = dreamed_obs[
                                    "pixels"
                                ]  # (n_steps * batch, C, H, W)
                                C, H, W = pixels_flat.shape[1:]
                                last_dreamed_pixels = torch.sigmoid(
                                    pixels_flat.view(n_steps, batch_sz, C, H, W)
                                ).detach()
                                with open("/tmp/claude/dream_debug.log", "a") as f:
                                    f.write(
                                        f"DECODER step={self.train_step}: shape={last_dreamed_pixels.shape}\n"
                                    )
                            except Exception as e:
                                with open("/tmp/claude/dream_debug.log", "a") as f:
                                    f.write(
                                        f"DECODER step={self.train_step}: EXCEPTION: {e}\n"
                                    )

                # Accumulate losses
                if total_wm_loss is None:
                    total_wm_loss = wm_loss
                    if self.train_step >= self.actor_warmup_steps:
                        total_actor_loss = actor_loss
                        total_critic_loss = critic_loss
                    else:
                        total_actor_loss = torch.tensor(
                            0.0, device=self.device, requires_grad=False
                        )
                        total_critic_loss = torch.tensor(
                            0.0, device=self.device, requires_grad=False
                        )
                else:
                    total_wm_loss = total_wm_loss + wm_loss  # type: ignore
                    if self.train_step >= self.actor_warmup_steps:
                        total_actor_loss = total_actor_loss + actor_loss  # type: ignore
                        total_critic_loss = total_critic_loss + critic_loss  # type: ignore

            # --- PROFILE: End forward, start backward ---
            torch.cuda.synchronize()
            timing.log_phase("forward", time.perf_counter() - t0)
            t0 = time.perf_counter()

            # Backprop
            assert (
                total_wm_loss is not None
                and total_actor_loss is not None
                and total_critic_loss is not None
            )

            if self.train_step >= self.actor_warmup_steps:
                # Full training: WM + critic + actor
                # Note: retain_graph not needed since dream states are detached
                # and each loss has an independent computation graph
                total_wm_loss.backward()
                total_critic_loss.backward()
                total_actor_loss.backward()
                self.wm_optimizer.step()
                self.critic_optimizer.step()
                self.actor_optimizer.step()
            else:
                # Warmup: WM only
                total_wm_loss.backward()
                self.wm_optimizer.step()
                # Log warmup progress (per-step loss)
                if self.train_step % 100 == 0:
                    seq_len = self.states.shape[1] if hasattr(self, "states") else 1
                    print(
                        f"Warmup {self.train_step}/{self.actor_warmup_steps} | WM Loss/step: {total_wm_loss.item() / seq_len:.4f}"
                    )

            # --- PROFILE: End backward ---
            torch.cuda.synchronize()
            timing.log_phase("backward", time.perf_counter() - t0)

            # --- PROFILE: Print summary ---
            timing.maybe_print(self.train_step)

            log_step = self.train_step
            self.train_step += 1

            # Log metrics to MLflow
            sequence_length = self.states.shape[1] if hasattr(self, "states") else 0
            self.log_metrics(
                total_wm_loss,
                total_actor_loss,
                total_critic_loss,
                wm_loss_components,
                sequence_length,
                dreamed_rewards_list,
                dreamed_values_list,
                actor_entropy_list,
                last_obs_pixels,
                last_obs_pixels_original,
                last_reconstruction_pixels,
                last_posterior_probs,
                last_dreamed_pixels,
                log_step,
            )

            # Send models to collector (skip in bootstrap mode - keep random actions)
            if self.mode != "bootstrap":
                if (
                    self.train_step >= self.actor_warmup_steps
                    and self.train_step % self.steps_per_weight_sync == 0
                ):
                    if self.train_step == self.actor_warmup_steps:
                        print(
                            f"Trainer: Warmup complete at step {self.train_step}. Sending initial models."
                        )
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
                # Log avg episode length for convergence tracking (from real episodes)
                avg_ep_len = self.replay_buffer.recent_avg_episode_length
                if avg_ep_len > 0:
                    self.logger.add_scalar(
                        "env/episode_length", avg_ep_len, self.train_step
                    )
                self.last_log_time = time.time()
                self.steps_since_log = 0

            # Checkpoint every N steps
            if self.train_step > 0 and self.train_step % self.checkpoint_interval == 0:
                if self.mode == "bootstrap":
                    self.save_wm_only_checkpoint()
                else:
                    self.save_checkpoint()

            profiler.step()

        profiler.__exit__(None, None, None)

        # Final save
        if self.mode == "bootstrap":
            self.save_wm_only_checkpoint(final=True)
            print(f"Bootstrap complete. WM checkpoint saved to {self.checkpoint_dir}")
        else:
            self.save_checkpoint(final=True)
            print(f"Training complete. Final checkpoint saved to {self.checkpoint_dir}")

        # Cleanup
        self.replay_buffer.stop()
        self.logger.close()

    def update_return_scale(self, lambda_returns, decay=0.99):
        flat = lambda_returns.detach().reshape(-1)
        range_batch = (
            torch.quantile(flat, 0.95).item() - torch.quantile(flat, 0.05).item()
        )
        self.S = self.S * decay + range_batch * (1 - decay)

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
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "mlflow_run_id": self.mlflow_run_id,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def save_wm_only_checkpoint(self, final=False):
        """Save WM-only checkpoint for bootstrap phase (no actor/critic)."""
        suffix = "final" if final else f"step_{self.train_step}"
        checkpoint = {
            "step": self.train_step,
            "encoder": self._get_model(self.encoder).state_dict(),
            "world_model": {
                k: v
                for k, v in self._get_model(self.world_model).state_dict().items()
                if k not in ("h_prev", "z_prev")
            },
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "checkpoint_type": "wm_only",
            "mlflow_run_id": self.mlflow_run_id,
        }
        path = os.path.join(self.checkpoint_dir, f"wm_checkpoint_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"WM-only checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path, reset_ac=False):
        """
        Load checkpoint with explicit control over actor/critic loading.

        Args:
            checkpoint_path: Path to checkpoint file
            reset_ac: If True, skip loading actor/critic (keep random init)

        Returns:
            checkpoint_type: 'wm_only', 'full', or 'reset_ac'
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load encoder and world_model (always present)
        self._get_model(self.encoder).load_state_dict(checkpoint["encoder"])
        self._get_model(self.world_model).load_state_dict(
            checkpoint["world_model"], strict=False
        )

        # Restore WM optimizer if present
        if "wm_optimizer" in checkpoint:
            self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])

        # Handle actor/critic based on explicit user intent
        has_ac = "actor" in checkpoint

        if reset_ac:
            # User explicitly requested fresh actor/critic
            print(f"Loaded WM weights from {checkpoint_path}")
            print("Actor/critic reset to random (--reset-ac)")
            return "reset_ac"
        elif not has_ac:
            # WM-only checkpoint, no AC to load
            print(f"Loaded WM-only checkpoint from {checkpoint_path}")
            print("Actor/critic initialized randomly")
            return "wm_only"
        else:
            # Full checkpoint with --resume: load everything
            self._get_model(self.actor).load_state_dict(checkpoint["actor"])
            self._get_model(self.critic).load_state_dict(checkpoint["critic"])

            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

            self.train_step = checkpoint.get("step", 0)
            print(
                f"Resumed full checkpoint from {checkpoint_path} at step {self.train_step}"
            )
            return "full"

    def dream_sequence(self, initial_h, initial_z_embed, num_dream_steps):
        """
        Generates a sequence of dreamed states and actions starting from an initial state.
        """
        dreamed_recurrent_states = []
        dreamed_actions_logits = []
        dreamed_actions_sampled = []

        # Treat the world model rollout as a fixed simulator for actor/critic updates.
        # We detach the starting state and all subsequent transitions so gradients do
        # not flow back into the world model when optimizing the policy or value head.
        dream_h = initial_h.detach()
        dream_z_embed = initial_z_embed.detach()

        for _ in range(num_dream_steps):
            dreamed_recurrent_states.append(dream_h.detach())
            action_logits = self.actor(dream_h.detach())
            dreamed_actions_logits.append(action_logits)

            # action_dist = torch.distributions.Categorical(logits=action_logits)
            action_dist = torch.distributions.Categorical(
                logits=action_logits, validate_args=False
            )
            action_sample = action_dist.sample()
            dreamed_actions_sampled.append(action_sample)
            action_onehot = F.one_hot(action_sample, num_classes=self.n_actions).float()

            # 1. Step the dynamics model to get the next h and prior_z
            dream_h_dyn, dream_prior_logits = self.world_model.step_dynamics(
                dream_z_embed, action_onehot, dream_h
            )

            # 2. Sample z from the prior with unimix (DreamerV3 Section 4)
            dream_prior_logits_mixed = unimix_logits(dream_prior_logits, unimix_ratio=0.01)
            dream_prior_dist = dist.Categorical(
                logits=dream_prior_logits_mixed, validate_args=False
            )
            dream_z_sample_indices = dream_prior_dist.sample()
            dream_z_sample = F.one_hot(
                dream_z_sample_indices, num_classes=self.d_hidden // 16
            ).float()

            # 3. Form the full state (h, z) for the next iteration's predictions
            dream_h = self.world_model.join_h_and_z(
                dream_h_dyn, dream_z_sample
            ).detach()
            dream_z_embed = self.world_model.z_embedding(
                dream_z_sample.view(dream_z_sample.size(0), -1)
            ).detach()

        # Stack the collected dreamed states and actions
        return (
            torch.stack(dreamed_recurrent_states),
            torch.stack(dreamed_actions_logits),
            torch.stack(dreamed_actions_sampled),
        )

    def calculate_lambda_returns(
        self,
        dreamed_rewards,
        dreamed_values,
        dreamed_continues,
        gamma,
        lam,
        num_dream_steps,
    ):
        """
        Calculates lambda-returns for a dreamed trajectory.
        """
        lambda_returns = []
        next_lambda_return = dreamed_values[-1]

        # Iterate backwards through the trajectory
        for i in reversed(range(num_dream_steps)):
            reward_t = dreamed_rewards[i]
            continue_prob_t = torch.sigmoid(dreamed_continues[i])
            value_t = dreamed_values[i]

            next_lambda_return = reward_t + gamma * continue_prob_t * (
                (1 - lam) * value_t + lam * next_lambda_return
            )
            lambda_returns.append(next_lambda_return)

        # The returns are calculated backwards, so we reverse them
        return torch.stack(lambda_returns).flip(0)

    def update_wm_loss(
        self,
        obs_reconstruction,
        obs_t,
        reward_dist,
        reward_t,
        terminated_t,
        continue_logits,
        posterior_dist,
        prior_logits,
    ):
        # Observation vectors use symlog squared loss
        obs_pred = symlog(obs_reconstruction["state"])
        obs_target = symlog(obs_t["state"])  # loss in symlog space
        beta_dyn = self.config.train.beta_dyn
        beta_rep = self.config.train.beta_rep
        beta_pred = self.config.train.beta_pred

        # There are three loss terms:
        # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
        # a. dynamics representation
        # -ln p(x|z,h) is trained with symlog squared loss
        pred_loss_vector = 1 / 2 * (obs_pred - obs_target) ** 2
        pred_loss_vector = pred_loss_vector.mean()

        # Pixel loss (only when using pixels)
        if self.use_pixels and "pixels" in obs_reconstruction and "pixels" in obs_t:
            pixel_probs = obs_reconstruction["pixels"]
            pixel_target = obs_t["pixels"]
            bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
            pred_loss_pixel = bce_with_logits_loss_fn(
                input=pixel_probs, target=pixel_target / 255.0
            )
        else:
            pred_loss_pixel = torch.tensor(0.0, device=self.device)

        reward_target = twohot_encode(reward_t, self.B)
        # Use soft cross-entropy for soft targets (twohot encoding)
        # reward_dist should be logits, reward_target is probabilities
        reward_loss = -torch.sum(
            reward_target * F.log_softmax(reward_dist, dim=-1), dim=-1
        ).mean()

        # c. continue predictor
        # The target is 1 if we continue, 0 if we terminate.
        continue_target = (1.0 - terminated_t.float()).unsqueeze(-1)
        pred_loss_continue = nn.BCEWithLogitsLoss()(continue_logits, continue_target)

        # Prediction loss is the sum of the individual losses
        l_pred = pred_loss_pixel + pred_loss_vector + reward_loss + pred_loss_continue

        # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
        # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
        # Log-likelihoods. Torch accepts logits

        # The "free bits" technique provides a minimum budget for the KL divergence.
        # prior_dist = dist.Categorical(logits=prior_logits)
        free_bits = 1.0
        # l_dyn_raw = dist.kl_divergence(
        #     dist.Categorical(logits=posterior_dist.logits.detach()),
        #     prior_dist,
        # ).mean()
        # l_rep_raw = dist.kl_divergence(
        #     posterior_dist,
        #     dist.Categorical(logits=prior_dist.logits.detach()),
        # ).mean()
        # Manual categorical KL to avoid distribution overhead.
        # Apply unimix to prior_logits for consistency (posterior already has unimix)
        prior_logits_mixed = unimix_logits(prior_logits, unimix_ratio=0.01)
        posterior_logits_detached = posterior_dist.logits.detach()
        log_posterior_detached = F.log_softmax(posterior_logits_detached, dim=-1)
        log_prior = F.log_softmax(prior_logits_mixed, dim=-1)
        posterior_probs_detached = log_posterior_detached.exp()
        l_dyn_raw = (
            (posterior_probs_detached * (log_posterior_detached - log_prior))
            .sum(dim=-1)
            .mean()
        )

        log_posterior = F.log_softmax(posterior_dist.logits, dim=-1)
        log_prior_detached = F.log_softmax(prior_logits_mixed.detach(), dim=-1)
        posterior_probs = log_posterior.exp()
        l_rep_raw = (
            (posterior_probs * (log_posterior - log_prior_detached)).sum(dim=-1).mean()
        )

        # Straight-through estimator for free bits:
        # Forward: loss = max(free_bits, raw) for reporting/scaling
        # Backward: gradient flows through raw value (not killed by max)
        l_dyn = l_dyn_raw + (free_bits - l_dyn_raw).clamp(min=0).detach()
        l_rep = l_rep_raw + (free_bits - l_rep_raw).clamp(min=0).detach()

        total_loss = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

        # Return both total loss and individual components for logging
        loss_dict = {
            "prediction_pixel": pred_loss_pixel,
            "prediction_vector": pred_loss_vector,
            "prediction_reward": reward_loss,
            "prediction_continue": pred_loss_continue,
            "dynamics": l_dyn,
            "representation": l_rep,
            "kl_dynamics_raw": l_dyn_raw,
            "kl_representation_raw": l_rep_raw,
        }

        return total_loss, loss_dict

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
            # All losses normalized to per-step for fair comparison
            if sequence_length > 0:
                norm = 1.0 / sequence_length
                beta_pred = self.config.train.beta_pred
                beta_dyn = self.config.train.beta_dyn
                beta_rep = self.config.train.beta_rep

                # Convert tensor loss components to CPU floats (single sync for all 8 components)
                wm_components_cpu = {k: v.item() for k, v in wm_loss_components.items()}

                # Per-step totals for each model
                wm_per_step = total_wm_loss.item() * norm
                self.logger.add_scalar("loss/wm/total", wm_per_step, step)
                self.logger.add_scalar(
                    "loss/actor/total", total_actor_loss.item() * norm, step
                )
                self.logger.add_scalar(
                    "loss/critic/total", total_critic_loss.item() * norm, step
                )

                # Raw component values (per-step, unscaled)
                pixel = wm_components_cpu["prediction_pixel"] * norm
                state = wm_components_cpu["prediction_vector"] * norm
                reward = wm_components_cpu["prediction_reward"] * norm
                cont = wm_components_cpu["prediction_continue"] * norm
                dyn = wm_components_cpu["dynamics"] * norm
                rep = wm_components_cpu["representation"] * norm

                # World model sub-component losses (grouped by module)
                # Decoder losses (reconstruction)
                self.logger.add_scalar("wm/decoder/pixel_loss", pixel, step)
                self.logger.add_scalar("wm/decoder/state_loss", state, step)
                # Predictor head losses
                self.logger.add_scalar("wm/reward_head/loss", reward, step)
                self.logger.add_scalar("wm/continue_head/loss", cont, step)
                # RSSM KL losses (after free bits)
                self.logger.add_scalar("wm/rssm/kl_dynamics", dyn, step)
                self.logger.add_scalar("wm/rssm/kl_representation", rep, step)

                # Scaled contributions to total (these should sum to loss/wm/total)
                pred_total = pixel + state + reward + cont
                self.logger.add_scalar(
                    "wm/scaled/prediction", beta_pred * pred_total, step
                )
                self.logger.add_scalar("wm/scaled/dynamics", beta_dyn * dyn, step)
                self.logger.add_scalar("wm/scaled/representation", beta_rep * rep, step)

                # Raw KL divergences (before free bits clipping, for debugging)
                self.logger.add_scalar(
                    "wm/rssm/kl_dynamics_raw",
                    wm_components_cpu["kl_dynamics_raw"] * norm,
                    step,
                )
                self.logger.add_scalar(
                    "wm/rssm/kl_representation_raw",
                    wm_components_cpu["kl_representation_raw"] * norm,
                    step,
                )
            else:
                # Fallback if no sequence
                self.logger.add_scalar("loss/wm/total", total_wm_loss.item(), step)
                self.logger.add_scalar("loss/actor/total", total_actor_loss.item(), step)
                self.logger.add_scalar("loss/critic/total", total_critic_loss.item(), step)

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
                self.logger.add_scalar(
                    "actor/entropy/std", entropy_stats[1].item(), step
                )

            # Learning rates
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
                self.logger.add_images("viz/wm/dream_strip", dream_strip.unsqueeze(0), step)

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
            while not self.model_queue.empty():
                self.model_queue.get_nowait()
            self.model_queue.put_nowait(models_to_send)
        except Full:
            print("Trainer: Model queue was full. Skipping update.")
            pass


def train_world_model(
    config,
    data_queue,
    model_queue,
    log_dir,
    checkpoint_path=None,
    mode="train",
    reset_ac=False,
    mlflow_run_id=None,
):
    # Set MLflow tracking URI for this child process
    if mlflow_run_id:
        mlruns_dir = os.path.join(os.path.dirname(log_dir), "mlruns")
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    trainer = WorldModelTrainer(
        config, data_queue, model_queue, log_dir, checkpoint_path, mode, reset_ac, mlflow_run_id
    )
    trainer.train_models()
