import math
import copy
import json
from dataclasses import asdict
from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F
from queue import Full
import mlflow
import os
import time

from .logging import create_step_metrics, log_step_metrics, log_progress
from .forward import dreamer_step
from ..runtime.replay_buffer import EpisodeReplayBuffer
from ..models import (
    symlog,
    symexp,
    resize_pixels_to_target,
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    LaProp,
    adaptive_gradient_clipping,
)
from .mlflow_logger import MLflowLogger
from ..runtime.env import create_env


class EnvData(NamedTuple):
    """Immutable batch of environment data sampled from replay."""

    states: torch.Tensor  # (B, T, n_obs) — symlog'd
    actions: torch.Tensor  # (B, T, n_actions)
    rewards: torch.Tensor  # (B, T)
    is_last: torch.Tensor  # (B, T)
    is_terminal: torch.Tensor  # (B, T)
    mask: torch.Tensor  # (B, T) — 1=real, 0=padded
    pixels: Optional[torch.Tensor] = None  # (B, T, C, H, W)
    pixels_original: Optional[torch.Tensor] = None  # (B, T, C, H, W)


class WorldModelTrainer:
    """Orchestrates DreamerV3 training: data loading, forward pass, optimization, logging.

    Owns all models (encoder, world model, actor, critic), optimizers,
    the replay buffer, and the MLflow logger. The actual algorithm lives
    in forward.dreamer_step(); this class handles the outer loop.
    """

    def __init__(
        self,
        config,
        data_queue,
        model_queue,
        log_dir,
        checkpoint_path=None,
        mlflow_run_id=None,
        dry_run=False,
        device="cpu",
    ):
        # Passed in args
        self.config = config  # Store config for use in methods
        self.dry_run = dry_run
        self.device = torch.device(device)
        self.use_pixels = config.use_pixels

        # EMA For Actor Critic
        self.S = 0.0
        self.ret_lo = None
        self.ret_hi = None
        self.retnorm_rate = 0.01

        # Model Init
        self.actor = initialize_actor(self.device, config)
        self.critic = initialize_critic(self.device, config)
        self.encoder, self.world_model = initialize_world_model(
            self.device, config, batch_size=config.batch_size
        )

        b_start = config.b_start
        b_end = config.b_end
        num_bins = int(getattr(self.critic.mlp[-1], "out_features"))
        beta_range = torch.linspace(
            start=b_start,
            end=b_end,
            steps=num_bins,
            device=self.device,
            dtype=torch.float32,
        )
        self.B = symexp(beta_range)

        if config.compile_models and hasattr(torch, "compile"):
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

        self.train_step = 0
        self.data_queue = data_queue
        self.model_queue = model_queue

        # Replay buffer: background thread drains queue, sample() returns instantly
        self.replay_buffer = EpisodeReplayBuffer(
            data_queue=data_queue,
            max_episodes=config.replay_buffer_size,
            min_episodes=config.min_buffer_episodes,
            sequence_length=config.sequence_length,
        )

        self.batch_size = config.batch_size  # needed by get_data_from_buffer
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
        if self.log_profile == "full":
            self.log_every = 10
            self.image_log_every = 100
        else:
            # Lean mode still needs responsive MLflow curves on slow Atari runs.
            self.log_every = 25
            self.image_log_every = 250
        self._wm_ac_counter = 0  # Counts WM steps since last AC step
        self._ac_training_started = False

        # Checkpointing
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save config snapshot
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Timing
        self.train_start_time = time.time()
        self.last_log_time = time.time()
        self.steps_since_log = 0
        self._resume_env_steps_offset = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            denom = max(1e-8, config.replay_ratio)
            self._resume_env_steps_offset = int(
                self.train_step
                * config.batch_size
                * config.sequence_length
                * config.action_repeat
                / denom
            )

    def get_data_from_buffer(self) -> EnvData | None:
        """Sample batch from replay buffer. Returns EnvData or None if empty."""
        raw_batch = self.replay_buffer.sample(self.batch_size)

        batch_pixels = []
        batch_pixels_original = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_is_last = []
        batch_is_terminal = []
        batch_mask = []

        for pixels, states, actions, rewards, is_last, is_terminal, mask in raw_batch:
            if self.use_pixels and pixels is not None:
                target_size = self.config.encoder_cnn_target_size
                pixels_tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                batch_pixels_original.append(pixels_tensor)
                if pixels_tensor.shape[-2:] != target_size:
                    pixels_tensor = resize_pixels_to_target(pixels_tensor, target_size)
                batch_pixels.append(pixels_tensor)

            batch_states.append(torch.from_numpy(states))
            batch_actions.append(torch.from_numpy(actions))
            batch_rewards.append(torch.from_numpy(rewards))
            batch_is_last.append(torch.from_numpy(is_last))
            batch_is_terminal.append(torch.from_numpy(is_terminal))
            batch_mask.append(torch.from_numpy(mask))

        if self.use_pixels and batch_pixels:
            pixels_out = torch.stack(batch_pixels).to(self.device).float()
            pixels_original_out = (
                torch.stack(batch_pixels_original).to(self.device).float()
            )
        else:
            pixels_out = None
            pixels_original_out = None

        # wonder if we should not do this here...
        states_out = symlog(torch.stack(batch_states).to(self.device))

        return EnvData(
            states=states_out,
            actions=torch.stack(batch_actions).to(self.device),
            rewards=torch.stack(batch_rewards).to(self.device),
            is_last=torch.stack(batch_is_last).to(self.device),
            is_terminal=torch.stack(batch_is_terminal).to(self.device),
            mask=torch.stack(batch_mask).to(self.device),
            pixels=pixels_out,
            pixels_original=pixels_original_out,
        )

    def prevent_stale_training(self):
        """Gate training speed to match environment data collection rate.

        Returns True (and sleeps briefly) if we've trained faster than
        the replay ratio allows, preventing overfitting to stale data.
        """
        # Replay ratio gating: wait if we've trained too fast relative to env steps
        env_steps = self.replay_buffer.total_env_steps + self._resume_env_steps_offset
        effective_burn_in = min(
            self.config.replay_burn_in,
            self.config.sequence_length - 1,
        )
        effective_seq_for_gate = max(1, self.config.sequence_length - effective_burn_in)
        target_train_steps = int(
            env_steps
            * self.config.replay_ratio
            / (self.batch_size * effective_seq_for_gate * self.config.action_repeat)
        )
        if self.train_step >= target_train_steps and env_steps > 0:
            time.sleep(0.01)  # Brief wait for more data
            return True
        return False

    def zeroize_state(self, actual_batch_size, h_dim):
        """Reset RSSM hidden state and zero all optimizer gradients for a new batch."""
        self.world_model.h_prev = torch.zeros(
            actual_batch_size, h_dim, device=self.device
        )
        self.world_model.z_prev = torch.zeros(
            actual_batch_size,
            self.config.num_latents,
            self.config.d_hidden // 16,
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
        return total_wm_loss, total_actor_loss, total_critic_loss

    def train_models(self):
        """Main training loop: sample → forward → backward → log → eval → checkpoint."""
        # Push current weights immediately so collectors do not spend startup
        # episodes in random-action mode when resuming from checkpoint.
        self.send_models_to_collector(self.train_step)

        while self.train_step < self.config.max_train_steps:
            if self.prevent_stale_training():
                continue
            if not self.replay_buffer.is_ready:
                time.sleep(0.01)
                continue

            # Apply cosine LR schedule (if enabled)
            if self.config.lr_cosine_decay:
                self.apply_lr_schedule()

            # Data loading
            batch = self.get_data_from_buffer()
            B, T = batch.states.shape[:2]

            # Skip if no data was retrieved
            if batch is None or T == 0:
                print(f"Trainer: No data was retrieved at step {self.train_step}.")
                continue

            # Reset hidden states per trajectory - match actual input batch size
            h_dim = self.world_model.h_prev.shape[1]

            # Reset hidden states per trajectory - match actual input batch size
            total_wm_loss, total_actor_loss, total_critic_loss = self.zeroize_state(
                B, h_dim
            )

            do_log_images = self.train_step % self.image_log_every == 0
            metrics = create_step_metrics(self.device, do_log_images)

            # Initialize loss variables in case loop doesn't execute
            t0 = time.perf_counter()

            total_wm_loss = None
            total_actor_loss = None
            total_critic_loss = None

            train_start_t = min(self.config.replay_burn_in, T - 1)
            effective_train_steps = T - train_start_t

            states_flat = batch.states.view(
                B * T, batch.states.shape[-1]
            )  # (B*T, n_obs)

            if self.use_pixels and batch.pixels is not None:
                pixels_flat = batch.pixels.view(
                    B * T, *batch.pixels.shape[2:]
                )  # (B*T, C, H, W)
                encoder_input = {"pixels": pixels_flat, "state": states_flat}
            else:
                encoder_input = states_flat  # State-only mode

            all_tokens = self.encoder(encoder_input)  # (B*T, token_dim)
            all_tokens = all_tokens.view(B, T, -1)

            # Decide AC update once per batch (ratio)
            skip_ac_batch = self.should_skip_ac_update()

            # Forward pass: RSSM rollout + dreaming + replay grounding
            result = dreamer_step(
                encoder=self.encoder,
                world_model=self.world_model,
                actor=self.actor,
                critic=self.critic,
                critic_ema=self.critic_ema,
                batch=batch,
                metrics=metrics,
                all_tokens=all_tokens,
                B=B,
                T=T,
                train_start_t=train_start_t,
                skip_ac=skip_ac_batch,
                bins=self.B,
                return_scale=self.S,
                config=self.config,
                device=self.device,
                use_pixels=self.use_pixels,
                do_log_images=do_log_images,
            )
            total_wm_loss = result.total_wm_loss
            total_actor_loss = result.total_actor_loss
            total_critic_loss = result.total_critic_loss
            metrics = result.metrics

            # Update return normalization EMA
            if result.last_lambda_returns is not None:
                self.update_return_scale(result.last_lambda_returns)

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

            # Use the batch-level AC skip decision
            if skip_ac_batch:
                # WM-only update this step
                total_wm_loss.backward()
                adaptive_gradient_clipping(self.wm_params)
                self.wm_optimizer.step()
            else:
                # Full training: WM + critic + actor
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
                        param_ema.data.mul_(self.config.critic_ema_decay).add_(
                            param.data, alpha=1 - self.config.critic_ema_decay
                        )

            # End backward

            log_step = self.train_step
            self.train_step += 1

            # Log metrics to MLflow
            log_step_metrics(
                self.logger,
                metrics,
                total_wm_loss,
                total_actor_loss,
                total_critic_loss,
                sequence_length=effective_train_steps,
                step=log_step,
                config=self.config,
                has_pixel_obs=self._has_pixel_obs,
                has_vector_obs=self._has_vector_obs,
                log_every=self.log_every,
                image_log_every=self.image_log_every,
                log_profile=self.log_profile,
                wm_optimizer=self.wm_optimizer,
                actor_optimizer=self.actor_optimizer,
                critic_optimizer=self.critic_optimizer,
                wm_ac_ratio_cosine=self.config.wm_ac_ratio_cosine,
                get_current_wm_ac_ratio=self.get_current_wm_ac_ratio,
            )

            # Send models to collector periodically
            if self.train_step % self.config.steps_per_weight_sync == 0:
                self.send_models_to_collector(self.train_step)

            # Periodic progress logging
            self.steps_since_log += 1
            if self.train_step % 100 == 0:
                elapsed = time.time() - self.last_log_time
                sps = self.steps_since_log / elapsed if elapsed > 0 else 0
                log_progress(
                    self.logger,
                    step=self.train_step,
                    max_steps=self.config.max_train_steps,
                    total_wm_loss=total_wm_loss,
                    total_actor_loss=total_actor_loss,
                    total_critic_loss=total_critic_loss,
                    seq_len=effective_train_steps,
                    steps_per_sec=sps,
                    env_steps=self.replay_buffer.total_env_steps + self._resume_env_steps_offset,
                    episodes_added=self.replay_buffer.total_episodes_added,
                    avg_ep_len=self.replay_buffer.recent_avg_episode_length,
                    elapsed_total=time.time() - self.train_start_time,
                )
                self.last_log_time = time.time()
                self.steps_since_log = 0

            if (
                self.config.eval_every > 0
                and self.config.eval_episodes > 0
                and self.train_step > 0
                and self.train_step % self.config.eval_every == 0
            ):
                eval_avg_len = self.evaluate_policy(
                    self.config.eval_episodes, step=self.train_step
                )
                early_stop = getattr(self.config, "early_stop_ep_length", 0)
                if early_stop > 0 and eval_avg_len >= early_stop:
                    print(f"SOLVED! eval_avg_len={eval_avg_len:.1f} >= {early_stop}")
                    break

            # Checkpoint every N steps (skip in dry_run)
            if (
                not self.dry_run
                and self.train_step > 0
                and self.train_step % self.config.checkpoint_interval == 0
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

    def should_skip_ac_update(self) -> bool:
        """Decide if we should skip the Actor-Critic update this batch based on WM:AC ratio."""
        self._wm_ac_counter += 1
        if self._wm_ac_counter < self.get_current_wm_ac_ratio():
            return True
        else:
            self._wm_ac_counter = 0
            return False

    def update_return_scale(self, lambda_returns, decay=0.99):
        """Update percentile-based return normalization EMA (DreamerV3 Appendix B)."""
        flat = lambda_returns.detach().reshape(-1)
        lo_batch = torch.quantile(flat, 0.05).item()
        hi_batch = torch.quantile(flat, 0.95).item()

        if self.ret_lo is None or self.ret_hi is None:
            self.ret_lo = lo_batch
            self.ret_hi = hi_batch
        else:
            rate = self.retnorm_rate
            self.ret_lo = (1.0 - rate) * self.ret_lo + rate * lo_batch
            self.ret_hi = (1.0 - rate) * self.ret_hi + rate * hi_batch

        self.S = max(1.0, float(self.ret_hi - self.ret_lo))

    def get_cosine_schedule(self, max_val: float, min_val: float) -> float:
        """Cosine schedule from max_val to min_val over training."""
        progress = min(1.0, self.train_step / max(1, self.config.max_train_steps))
        return min_val + 0.5 * (max_val - min_val) * (1 + math.cos(math.pi * progress))

    def get_current_wm_ac_ratio(self) -> int:
        """Get current WM:AC ratio (possibly scheduled)."""
        if not self.config.wm_ac_ratio_cosine:
            return self.config.wm_ac_ratio
        # Normal: max→min (8→2), Inverted: min→max (2→8)
        if self.config.wm_ac_ratio_invert:
            ratio = self.get_cosine_schedule(
                float(self.config.wm_ac_ratio_min), float(self.config.wm_ac_ratio_max)
            )
        else:
            ratio = self.get_cosine_schedule(
                float(self.config.wm_ac_ratio_max), float(self.config.wm_ac_ratio_min)
            )
        return max(1, round(ratio))

    def apply_lr_schedule(self):
        """Apply cosine LR decay to all optimizers."""
        scale = self.get_cosine_schedule(1.0, self.config.lr_cosine_min_factor)
        for pg in self.wm_optimizer.param_groups:
            pg["lr"] = self.config.wm_lr * scale
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = self.config.actor_lr * scale
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.config.critic_lr * scale

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
                action_onehot = torch.zeros(1, self.config.n_actions, device=self.device)
                z_prev = torch.zeros(
                    1, self.config.num_latents, self.config.d_hidden // 16, device=self.device
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

                    # Encoder returns tokens, posterior conditioned on h_t
                    tokens = self.encoder(encoder_input)
                    posterior_logits = self.world_model.compute_posterior(h, tokens)
                    z_indices = posterior_logits.argmax(dim=-1)
                    z_onehot = F.one_hot(
                        z_indices, num_classes=self.config.d_hidden // 16
                    ).float()
                    z_sample = z_onehot

                    actor_input = self.world_model.join_h_and_z(h, z_sample)
                    action_logits = self.actor(actor_input)
                    action = action_logits.argmax(dim=-1)
                    action_onehot = F.one_hot(
                        action, num_classes=self.config.n_actions
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
            "ret_lo": self.ret_lo,
            "ret_hi": self.ret_hi,
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
        self.ret_lo = checkpoint.get("ret_lo", self.ret_lo)
        self.ret_hi = checkpoint.get("ret_hi", self.ret_hi)

        self.train_step = checkpoint.get("step", 0)
        print(f"Resumed checkpoint from {checkpoint_path} at step {self.train_step}")


    def send_models_to_collector(self, step: int):
        """Push current model weights to the data collection process."""
        if self.model_queue is None:
            return

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
            print(f"Trainer: Sent models at step {step} (cleared {cleared} old)")
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
    device="cpu",
):
    """Entry point: construct a WorldModelTrainer and run the training loop."""
    # Set MLflow tracking URI and join existing run in this child process (skip in dry_run)
    if mlflow_run_id and not dry_run:
        mlruns_dir = os.path.abspath(os.path.join("runs", "mlruns"))
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
        device,
    )
    trainer.train_models()
