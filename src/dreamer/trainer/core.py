import math
import copy
import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F
from queue import Full
import mlflow
import numpy as np
import os
import time

from .logging import create_step_metrics, log_step_metrics, log_progress
from .forward import dreamer_step
from ..runtime.replay_buffer import EpisodeReplayBuffer, EnvData
from .checkpoints import save_checkpoint, load_checkpoint
from ..models import (
    symlog,
    symexp,
    resize_pixels_to_target,
    initialize_actor,
    initialize_critic,
    initialize_q_critic,
    initialize_world_model,
    LaProp,
    adaptive_gradient_clipping,
)
from .mlflow_logger import MLflowLogger
from ..runtime.env import create_env
from ..run_manifest import file_sha256, read_run_manifest, update_run_manifest


logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed process-local RNGs for reproducible long-run comparisons."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class EvaluationResult:
    """Deterministic evaluation metrics used for logging and checkpoint choice."""

    avg_length: float
    avg_reward: float
    win_rate: float

    def metric_value(self, metric: str) -> float:
        values = {
            "episode_length": self.avg_length,
            "episode_reward": self.avg_reward,
            "win_rate": self.win_rate,
        }
        try:
            return values[metric]
        except KeyError as exc:
            supported = ", ".join(sorted(values))
            raise ValueError(
                f"Unsupported eval_metric={metric!r}; choose one of: {supported}"
            ) from exc


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
        model_queues,
        log_dir,
        checkpoint_path=None,
        mlflow_run_id=None,
        dry_run=False,
        device="cpu",
    ):
        # Passed in args
        if data_queue is None:
            raise ValueError("WorldModelTrainer requires a collector data queue")
        self.config = config  # Store config for use in methods
        self.dry_run = dry_run
        self.device = torch.device(device)
        self.use_pixels = config.use_pixels
        self.log_dir = log_dir
        try:
            self.run_manifest_id = read_run_manifest(log_dir)["run_id"]
        except (FileNotFoundError, KeyError) as exc:
            raise RuntimeError(
                "training requires a run_manifest.json created by run_training"
            ) from exc
        seed_everything(int(getattr(config, "seed", 0)) + 2000)

        # EMA For Actor Critic
        self.S = 0.0
        self.ret_lo = None
        self.ret_hi = None
        self.retnorm_rate = 0.01

        # Model Init
        self.actor = initialize_actor(self.device, config)
        self.critic = initialize_critic(self.device, config)
        self.q_critic = initialize_q_critic(self.device, config)
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
                self.q_critic = torch.compile(self.q_critic)
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
        self.critic_params = list(self.critic.parameters()) + list(
            self.q_critic.parameters()
        )
        self.critic_optimizer = LaProp(
            self.critic_params,
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
        self.q_critic_ema = copy.deepcopy(self.q_critic)
        for param in self.q_critic_ema.parameters():
            param.requires_grad = False

        self.train_step = 0
        self.data_queue = data_queue
        self.model_queues = list(model_queues)

        # Replay buffer: background thread drains queue, sample() returns instantly
        self.replay_buffer = EpisodeReplayBuffer(
            data_queue=data_queue,
            max_episodes=config.replay_buffer_size,
            min_episodes=config.min_buffer_episodes,
            sequence_length=config.sequence_length,
            gamma=config.gamma,
        )

        self.batch_size = config.batch_size  # needed by get_data_from_buffer
        self.replay_buffer.start()
        print(f"Replay buffer started (max={config.replay_buffer_size} episodes)")

        # Initialize MLflow logger with the provided log directory
        self.mlflow_run_id = mlflow_run_id
        self.logger = MLflowLogger(
            log_dir=log_dir, run_id=mlflow_run_id, enabled=not dry_run
        )
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
        self.best_eval_score = None
        self.best_eval_step = None
        self.best_eval_metric = getattr(config, "eval_metric", "episode_reward")

        if checkpoint_path:
            chk = load_checkpoint(
                checkpoint_path,
                self.device,
                self.encoder,
                self.world_model,
                self.actor,
                self.critic,
                self.critic_ema,
                self.q_critic,
                self.q_critic_ema,
                self.wm_optimizer,
                self.actor_optimizer,
                self.critic_optimizer,
                self.S,
                self.ret_lo,
                self.ret_hi,
            )
            self.train_step = chk["step"]
            self.S = chk["return_scale"]
            self.ret_lo = chk["ret_lo"]
            self.ret_hi = chk["ret_hi"]
            self.best_eval_score = chk["best_eval_score"]
            self.best_eval_step = chk["best_eval_step"]
            checkpoint_metric = chk["best_eval_metric"]
            if checkpoint_metric is not None:
                self.best_eval_metric = checkpoint_metric
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
        batch_future_returns = []
        batch_mask = []

        for (
            pixels,
            states,
            actions,
            rewards,
            is_last,
            is_terminal,
            future_returns,
            mask,
        ) in raw_batch:
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
            batch_future_returns.append(torch.from_numpy(future_returns))
            batch_mask.append(torch.from_numpy(mask))

        if self.use_pixels and batch_pixels:
            pixels_out = torch.stack(batch_pixels).to(self.device).float()
            pixels_original_out = (
                torch.stack(batch_pixels_original).to(self.device).float()
            )
        else:
            pixels_out = None
            pixels_original_out = None

        states_out = torch.stack(batch_states).to(self.device)

        return EnvData(
            states=states_out,
            actions=torch.stack(batch_actions).to(self.device),
            rewards=torch.stack(batch_rewards).to(self.device),
            is_last=torch.stack(batch_is_last).to(self.device),
            is_terminal=torch.stack(batch_is_terminal).to(self.device),
            future_returns=torch.stack(batch_future_returns).to(self.device),
            mask=torch.stack(batch_mask).to(self.device),
            pixels=pixels_out,
            pixels_original=pixels_original_out,
        )

    def prevent_stale_training(self) -> bool:
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

    def train_models(self):
        """Main training loop: sample → forward → backward → log → eval → checkpoint."""
        # Keep fresh runs in random-action collection during WM warmup. Once
        # warmup is over, send policy weights so the collector switches to the
        # learned actor. Resumed checkpoints beyond warmup still sync at startup.
        if (
            self.train_step < self.config.max_train_steps
            and self.train_step >= self.config.actor_warmup_steps
        ):
            self.send_models_to_collectors(self.train_step)

        stop_reason = "max_train_steps"
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
            batch = self.replay_buffer.sample_tensors(
                self.batch_size,
                self.device,
                use_pixels=self.use_pixels,
                target_size=self.config.encoder_cnn_target_size,
            )
            B, T = batch.states.shape[:2]

            # Skip if no data was retrieved
            if batch is None or T == 0:
                print(f"Trainer: No data was retrieved at step {self.train_step}.")
                continue

            # Reset hidden states per trajectory - match actual input batch size
            self.world_model.init_state(B, self.device)

            self.wm_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            total_wm_loss = None
            total_actor_loss = None
            total_critic_loss = None

            do_log_images = self.train_step % self.image_log_every == 0
            metrics = create_step_metrics(self.device, do_log_images)

            # Initialize loss variables in case loop doesn't execute
            t0 = time.perf_counter()

            total_wm_loss = None
            total_actor_loss = None
            total_critic_loss = None

            train_start_t = min(self.config.replay_burn_in, T - 1)
            effective_train_steps = T - train_start_t

            states_flat_raw = batch.states.view(
                B * T, batch.states.shape[-1]
            )  # (B*T, n_obs)
            states_flat = symlog(states_flat_raw)

            if self.use_pixels and batch.pixels is not None:
                pixels_flat = batch.pixels.view(
                    B * T, *batch.pixels.shape[2:]
                )  # (B*T, C, H, W)
                encoder_input = {"pixels": pixels_flat, "state": states_flat}
            else:
                encoder_input = states_flat  # State-only mode

            all_tokens = self.encoder(encoder_input)  # (B*T, token_dim)
            all_tokens = all_tokens.view(B, T, -1)

            # Apply the WM:AC ratio to both heads, but freeze only the actor
            # during actor warmup. This gives the critic time to learn from the
            # random policy before its estimates can change that policy.
            in_actor_warmup = self.train_step < self.config.actor_warmup_steps
            skip_ac_batch = self.should_skip_ac_update()
            skip_actor_batch = in_actor_warmup or skip_ac_batch
            skip_critic_batch = skip_ac_batch

            # Forward pass: RSSM rollout + dreaming + replay grounding
            result = dreamer_step(
                encoder=self.encoder,
                world_model=self.world_model,
                actor=self.actor,
                critic=self.critic,
                critic_ema=self.critic_ema,
                q_critic=self.q_critic,
                q_critic_ema=self.q_critic_ema,
                batch=batch,
                metrics=metrics,
                all_tokens=all_tokens,
                B=B,
                T=T,
                train_start_t=train_start_t,
                skip_actor=skip_actor_batch,
                skip_critic=skip_critic_batch,
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

            total_wm_loss.backward()
            has_critic_grad = bool(total_critic_loss.requires_grad)
            has_actor_grad = bool(total_actor_loss.requires_grad)
            if has_critic_grad:
                total_critic_loss.backward()
            if has_actor_grad:
                total_actor_loss.backward()

            # AGC: clip gradients based on param/grad norm ratio (DreamerV3).
            # Use more aggressive clipping for pixel observations (prevent NaN).
            agc_clip = 0.15 if self.use_pixels else 0.3
            adaptive_gradient_clipping(self.wm_params, clip_factor=agc_clip)
            self.wm_optimizer.step()
            if has_critic_grad:
                adaptive_gradient_clipping(self.critic_params, clip_factor=agc_clip)
                self.critic_optimizer.step()
            if has_actor_grad:
                adaptive_gradient_clipping(
                    self.actor.parameters(), clip_factor=agc_clip
                )
                self.actor_optimizer.step()

            # Polyak update for critic EMA whenever the critic trained.
            if has_critic_grad:
                with torch.no_grad():
                    for param, param_ema in zip(
                        self.critic.parameters(), self.critic_ema.parameters()
                    ):
                        param_ema.data.mul_(self.config.critic_ema_decay).add_(
                            param.data, alpha=1 - self.config.critic_ema_decay
                        )
                    for param, param_ema in zip(
                        self.q_critic.parameters(), self.q_critic_ema.parameters()
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
            if (
                self.train_step < self.config.max_train_steps
                and self.train_step >= self.config.actor_warmup_steps
                and self.train_step % self.config.steps_per_weight_sync == 0
            ):
                self.send_models_to_collectors(self.train_step)

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
                eval_result = self.evaluate_policy(
                    self.config.eval_episodes, step=self.train_step
                )
                eval_score = eval_result.metric_value(self.best_eval_metric)
                is_new_best = (
                    self.best_eval_score is None
                    or eval_score > self.best_eval_score
                )
                if is_new_best:
                    self.best_eval_score = eval_score
                    self.best_eval_step = self.train_step
                    if not self.dry_run:
                        best_path = self.save_training_checkpoint(label="best")
                        update_run_manifest(
                            self.log_dir,
                            {
                                "artifacts": {
                                    "best_checkpoint": {
                                        "path": os.path.relpath(
                                            best_path, self.log_dir
                                        ),
                                        "sha256": file_sha256(best_path),
                                    }
                                }
                            },
                        )
                update_run_manifest(
                    self.log_dir,
                    {
                        "evaluation": {
                            "metric": self.best_eval_metric,
                            "latest_score": eval_score,
                            "latest_step": self.train_step,
                            "best_score": self.best_eval_score,
                            "best_step": self.best_eval_step,
                            "latest_metrics": {
                                "episode_length": eval_result.avg_length,
                                "episode_reward": eval_result.avg_reward,
                                "win_rate": eval_result.win_rate,
                            },
                        }
                    },
                )
                early_stop = getattr(self.config, "early_stop_ep_length", 0)
                if early_stop > 0 and eval_result.avg_length >= early_stop:
                    print(
                        f"SOLVED! eval_avg_len={eval_result.avg_length:.1f} "
                        f">= {early_stop}"
                    )
                    stop_reason = "early_stop"
                    break

            # Checkpoint every N steps (skip in dry_run)
            if (
                not self.dry_run
                and self.train_step > 0
                and self.train_step % self.config.checkpoint_interval == 0
            ):
                self.save_training_checkpoint()

        # Final save (skip in dry_run)
        if not self.dry_run:
            final_path = self.save_training_checkpoint(final=True)
            update_run_manifest(
                self.log_dir,
                {
                    "artifacts": {
                        "final_checkpoint": {
                            "path": os.path.relpath(final_path, self.log_dir),
                            "sha256": file_sha256(final_path),
                        }
                    }
                },
            )
            print(f"Training complete. Final checkpoint saved to {self.checkpoint_dir}")
        else:
            print("DRY RUN complete - no checkpoint saved.")

        update_run_manifest(
            self.log_dir,
            {
                "progress": {
                    "train_step": self.train_step,
                    "env_steps": self.replay_buffer.total_env_steps
                    + self._resume_env_steps_offset,
                },
                "evaluation": {
                    "best_score": self.best_eval_score,
                    "best_step": self.best_eval_step,
                    "metric": self.best_eval_metric,
                },
                "outcome": {"stop_reason": stop_reason},
            },
        )

        # Cleanup
        self.replay_buffer.stop()
        self.logger.close()

    def save_training_checkpoint(
        self, *, final: bool = False, label: str | None = None
    ) -> str:
        """Persist the complete trainer state using the shared checkpoint contract."""
        return save_checkpoint(
            self.checkpoint_dir,
            self.train_step,
            self.encoder,
            self.world_model,
            self.actor,
            self.critic,
            self.critic_ema,
            self.q_critic,
            self.q_critic_ema,
            self.wm_optimizer,
            self.actor_optimizer,
            self.critic_optimizer,
            self.S,
            self.ret_lo,
            self.ret_hi,
            self.mlflow_run_id,
            final=final,
            label=label,
            best_eval_score=self.best_eval_score,
            best_eval_step=self.best_eval_step,
            best_eval_metric=self.best_eval_metric,
            run_id=self.run_manifest_id,
        )

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

    def evaluate_policy(self, num_episodes: int, step: int) -> EvaluationResult:
        """Run deterministic evaluation episodes and log summary metrics.

        Returns:
            Aggregate deterministic evaluation metrics.
        """
        if num_episodes <= 0:
            return EvaluationResult(0.0, 0.0, 0.0)

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
            eval_seed = int(getattr(self.config, "seed", 0)) + 1_000_000 + step * 1000
            for episode_idx in range(num_episodes):
                obs, _info = env.reset(seed=eval_seed + episode_idx)
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

        return EvaluationResult(avg_len, avg_reward, win_rate)

    def _get_model(self, model):
        """Get underlying model (handles both compiled and non-compiled)."""
        return getattr(model, "_orig_mod", model)

    def send_models_to_collectors(self, step: int) -> None:
        """Publish one coherent weight snapshot to every collector mailbox."""
        # Exclude h_prev/z_prev buffers as they have batch-size-dependent shapes
        wm = self._get_model(self.world_model)
        actor = self._get_model(self.actor)
        encoder = self._get_model(self.encoder)

        # state_dict tensors alias live parameters on CPU. Clone them so every
        # collector receives one coherent training-step snapshot.
        wm_state = {
            k: v.detach().to(device="cpu", copy=True)
            for k, v in wm.state_dict().items()
            if k not in ("h_prev", "z_prev")
        }
        models_to_send = {
            "version": step,
            "actor": {
                k: v.detach().to(device="cpu", copy=True)
                for k, v in actor.state_dict().items()
            },
            "encoder": {
                k: v.detach().to(device="cpu", copy=True)
                for k, v in encoder.state_dict().items()
            },
            "world_model": wm_state,
        }

        sent = []
        pending = []
        for collector_id, model_queue in enumerate(self.model_queues):
            try:
                model_queue.put_nowait(models_to_send)
                sent.append(collector_id)
            except Full:
                # A full one-item mailbox means this collector already has an
                # unseen update. Do not block training behind a slow environment.
                pending.append(collector_id)

        log = logger.warning if pending else logger.info
        log(
            "model_weights_published version=%d delivered=%s pending=%s",
            step,
            sent,
            pending,
        )


def train_world_model(
    config,
    data_queue,
    model_queues,
    training_done_event,
    collectors_stopped_event,
    log_dir,
    checkpoint_path=None,
    mlflow_run_id=None,
    dry_run=False,
    device="cpu",
):
    """Entry point: construct a WorldModelTrainer and run the training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(name)s %(message)s",
    )
    # Set MLflow tracking URI and join existing run in this child process (skip in dry_run)
    if mlflow_run_id and not dry_run:
        mlruns_dir = os.path.abspath(os.path.join("runs", "mlruns"))
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
        # Join the existing run (subprocess doesn't inherit active run from parent)
        mlflow.start_run(run_id=mlflow_run_id)

    trainer = WorldModelTrainer(
        config,
        data_queue,
        model_queues,
        log_dir,
        checkpoint_path,
        mlflow_run_id,
        dry_run,
        device,
    )
    trainer.train_models()
    training_done_event.set()
    if not collectors_stopped_event.wait(timeout=15.0):
        raise TimeoutError(
            "trainer timed out waiting for collectors to finish shutdown"
        )
