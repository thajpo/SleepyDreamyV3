import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from queue import Full, Empty
from torch.utils.tensorboard import SummaryWriter
import os
import time

from .config import config
from .trainer_utils import (
    symlog,
    symexp,
    twohot_encode,
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    resize_pixels_to_target,
)
from .encoder import ObservationEncoder, ThreeLayerMLP
from .world_model import RSSMWorldModel
from datetime import datetime

class WorldModelTrainer:
    def __init__(
        self,
        config,
        data_queue,
        model_queue,
        log_dir,
        checkpoint_path=None,
        mode='train',
        reset_ac=False,
    ):
        self.device = torch.device(config.general.device)
        self.model_update_frequency = 10 # fix later
        self.n_dream_steps = config.train.num_dream_steps
        self.gamma = config.train.gamma
        self.lam = config.train.lam
        self.actor_entropy_coef = config.train.actor_entropy_coef
        self.normalize_advantages = config.train.normalize_advantages
        self.actor_warmup_steps = config.train.actor_warmup_steps
        b_start = config.train.b_start
        b_end = config.train.b_end
        beta_range = torch.arange(
        start=b_start,
        end=b_end,
        device=self.device,
        )
        self.B = symexp(beta_range)

        self.actor = initialize_actor(self.device)
        self.critic = initialize_critic(self.device)
        self.encoder, self.world_model = initialize_world_model(
            self.device,
            batch_size=config.train.batch_size
        )

        # JIT compile models for faster execution
        self.encoder = torch.compile(self.encoder)
        self.world_model = torch.compile(self.world_model)
        self.actor = torch.compile(self.actor)
        self.critic = torch.compile(self.critic)

        self.wm_params = list(self.encoder.parameters()) + list(self.world_model.parameters())
        self.wm_optimizer = optim.Adam(
            self.wm_params, lr=config.train.wm_lr, weight_decay=config.train.weight_decay)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.train.critic_lr, weight_decay=config.train.weight_decay)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.train.actor_lr, weight_decay=config.train.weight_decay)

        self.max_train_steps = config.train.max_train_steps
        self.train_step = 0
        self.data_queue = data_queue
        self.model_queue = model_queue
        self.d_hidden = config.models.d_hidden
        self.n_actions = config.environment.n_actions
        self.steps_per_weight_sync = config.train.steps_per_weight_sync
        self.batch_size = config.train.batch_size

        # Initialize TensorBoard writer with the provided log directory
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard logging to: {log_dir}")

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
            self.checkpoint_type = self.load_checkpoint(checkpoint_path, reset_ac=reset_ac)
        else:
            self.checkpoint_type = None

        # Mode-specific settings
        if mode == 'bootstrap':
            self.actor_warmup_steps = float('inf')  # Never train AC
            print("Bootstrap mode: WM-only training with random actions")
        elif mode == 'dreamer':
            self.actor_warmup_steps = 0  # Immediate AC training
            if reset_ac:
                print("Dreamer mode: Fresh actor/critic, keeping WM weights")
            else:
                print("Dreamer mode: Resuming all weights from checkpoint")


    def get_data_from_queue(self):
        """Collect batch_size sequences from queue, truncate to min length, stack into batch."""
        try:
            batch_pixels = []
            batch_pixels_original = []
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_terminated = []

            # Collect batch_size sequences
            for _ in range(self.batch_size):
                pixels, states, actions, rewards, terminated = self.data_queue.get()
                # Convert pixels from (T, H, W, C) to (T, C, H, W)
                pixels_tensor = torch.from_numpy(pixels).float().permute(0, 3, 1, 2)
                batch_pixels_original.append(pixels_tensor)  # Keep original resolution
                pixels_resized = resize_pixels_to_target(pixels_tensor, config.models.encoder.cnn.target_size)
                batch_pixels.append(pixels_resized)
                batch_states.append(torch.from_numpy(states))
                batch_actions.append(torch.from_numpy(actions))
                batch_rewards.append(torch.from_numpy(rewards))
                batch_terminated.append(torch.from_numpy(terminated))

            # Find minimum sequence length and truncate all to match
            min_len = min(p.shape[0] for p in batch_pixels)
            batch_pixels = [p[:min_len] for p in batch_pixels]
            batch_pixels_original = [p[:min_len] for p in batch_pixels_original]
            batch_states = [s[:min_len] for s in batch_states]
            batch_actions = [a[:min_len] for a in batch_actions]
            batch_rewards = [r[:min_len] for r in batch_rewards]
            batch_terminated = [t[:min_len] for t in batch_terminated]

            # Stack into batch tensors: (B, T, ...)
            self.pixels = torch.stack(batch_pixels).to(self.device)  # (B, T, C, H, W)
            self.pixels_original = torch.stack(batch_pixels_original).to(self.device)  # Original res
            self.states = symlog(torch.stack(batch_states).to(self.device))  # (B, T, state_dim)
            self.actions = torch.stack(batch_actions).to(self.device)  # (B, T, n_actions)
            self.rewards = torch.stack(batch_rewards).to(self.device)  # (B, T)
            self.terminated = torch.stack(batch_terminated).to(self.device)  # (B, T)
        except Empty:
            pass

    def train_models(self):
        while self.train_step < self.max_train_steps:
            self.get_data_from_queue()  # Collects batch_size sequences

            # Skip if no data was retrieved
            if not hasattr(self, 'pixels') or self.pixels.shape[1] == 0:
                print(f"Trainer: No data was retrieved at step {self.train_step}.")
                continue

            # Reset hidden states per trajectory - match actual input batch size
            actual_batch_size = self.pixels.shape[0]
            h_dim = self.world_model.h_prev.shape[1]
            self.world_model.h_prev = torch.zeros(actual_batch_size, h_dim, device=self.device)

            # Zero gradients before accumulating losses
            self.wm_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            # Initialize loss accumulators - will be set on first iteration
            total_wm_loss = None
            total_actor_loss = None
            total_critic_loss = None
            # Accumulate individual loss components for logging
            wm_loss_components = {
                'prediction_pixel': 0.0,
                'prediction_vector': 0.0,
                'prediction_reward': 0.0,
                'prediction_continue': 0.0,
                'dynamics': 0.0,
                'representation': 0.0,
                'kl_dynamics_raw': 0.0,
                'kl_representation_raw': 0.0,
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

            for t_step in range(self.pixels.shape[1]):  # shape[1] is time dimension in (1, T, C, H, W)
                # Extract time step with batch dimension: (1, T, C, H, W) -> (1, C, H, W)
                obs_t = {"pixels": self.pixels[:, t_step], "state": self.states[:, t_step]}
                action_t = self.actions[:, t_step]
                reward_t = self.rewards[:, t_step]
                terminated_t = self.terminated[:, t_step]

                posterior_logits = self.encoder(obs_t)  # This is z_t
                posterior_dist = dist.Categorical(logits=posterior_logits)

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
                    prior_logits
                    )
                
                # Accumulate individual loss components
                for key in wm_loss_components:
                    wm_loss_components[key] += wm_loss_dict[key].item()

                # Store visualization data
                last_obs_pixels = obs_t["pixels"]
                last_obs_pixels_original = self.pixels_original[:, t_step]  # Original resolution
                last_reconstruction_pixels = obs_reconstruction["pixels"]
                last_posterior_probs = posterior_dist.probs.detach()  # (batch, d_hidden, d_hidden/16)

                # --- Dream Sequence for Actor-Critic (skip during warmup) ---
                if self.train_step >= self.actor_warmup_steps:
                    h_prev_backup = self.world_model.h_prev.clone()
                    (
                        dreamed_recurrent_states,
                        dreamed_actions_logits,
                        dreamed_actions_sampled
                    ) = self.dream_sequence(
                        h_z_joined,
                        self.world_model.z_embedding(posterior_dist.probs.view(actual_batch_size, -1)),
                        self.n_dream_steps
                    )
                    self.world_model.h_prev = h_prev_backup

                    dreamed_rewards_logits = self.world_model.reward_predictor(
                        dreamed_recurrent_states
                    ).detach()
                    dreamed_rewards_probs = F.softmax(dreamed_rewards_logits, dim=-1)
                    dreamed_rewards = torch.sum(dreamed_rewards_probs * self.B, dim=-1).detach()
                    dreamed_rewards_list.append(dreamed_rewards.detach().cpu())

                    dreamed_continues = self.world_model.continue_predictor(
                        dreamed_recurrent_states
                    ).detach().squeeze(-1)  # Remove trailing (1,) dimension

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
                        self.n_dream_steps
                    )

                    actor_loss, critic_loss, entropy = self.update_actor_critic_losses(
                        dreamed_values_logits,
                        dreamed_values,
                        lambda_returns,
                        dreamed_actions_logits,
                        dreamed_actions_sampled
                    )
                    actor_entropy_list.append(entropy.detach().cpu())

                    # Decode dreamed states for visualization (every 50 steps)
                    if self.train_step % 50 == 0:
                        with torch.no_grad():
                            dreamed_obs = self.world_model.decoder(dreamed_recurrent_states)
                            last_dreamed_pixels = torch.sigmoid(dreamed_obs["pixels"]).detach()

                # Accumulate losses
                if total_wm_loss is None:
                    total_wm_loss = wm_loss
                    if self.train_step >= self.actor_warmup_steps:
                        total_actor_loss = actor_loss
                        total_critic_loss = critic_loss
                    else:
                        total_actor_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                        total_critic_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                else:
                    total_wm_loss = total_wm_loss + wm_loss  # type: ignore
                    if self.train_step >= self.actor_warmup_steps:
                        total_actor_loss = total_actor_loss + actor_loss  # type: ignore
                        total_critic_loss = total_critic_loss + critic_loss  # type: ignore

            # Backprop
            assert total_wm_loss is not None and total_actor_loss is not None and total_critic_loss is not None

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
                    seq_len = self.pixels.shape[1] if hasattr(self, 'pixels') else 1
                    print(f"Warmup {self.train_step}/{self.actor_warmup_steps} | WM Loss/step: {total_wm_loss.item()/seq_len:.4f}")

            self.train_step += 1

            # Log metrics to TensorBoard
            sequence_length = self.pixels.shape[1] if hasattr(self, 'pixels') else 0
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
                last_dreamed_pixels
            )

            # Send models to collector (skip in bootstrap mode - keep random actions)
            if self.mode != 'bootstrap':
                if self.train_step >= self.actor_warmup_steps and self.train_step % self.steps_per_weight_sync == 0:
                    if self.train_step == self.actor_warmup_steps:
                        print(f"Trainer: Warmup complete at step {self.train_step}. Sending initial models.")
                    self.send_models_to_collector(self.train_step)

            # Periodic logging (every 100 steps)
            self.steps_since_log += 1
            if self.train_step % 100 == 0:
                elapsed = time.time() - self.last_log_time
                steps_per_sec = self.steps_since_log / elapsed if elapsed > 0 else 0
                seq_len = sequence_length if sequence_length > 0 else 1
                eta_hours = (self.max_train_steps - self.train_step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0

                print(f"Step {self.train_step}/{self.max_train_steps} | "
                      f"{steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h | "
                      f"WM: {total_wm_loss.item()/seq_len:.4f} | "
                      f"Actor: {total_actor_loss.item()/seq_len:.4f} | "
                      f"Critic: {total_critic_loss.item()/seq_len:.4f}")

                self.writer.add_scalar("training/steps_per_sec", steps_per_sec, self.train_step)
                self.last_log_time = time.time()
                self.steps_since_log = 0

            # Checkpoint every N steps
            if self.train_step > 0 and self.train_step % self.checkpoint_interval == 0:
                if self.mode == 'bootstrap':
                    self.save_wm_only_checkpoint()
                else:
                    self.save_checkpoint()

        # Final save
        if self.mode == 'bootstrap':
            self.save_wm_only_checkpoint(final=True)
            print(f"Bootstrap complete. WM checkpoint saved to {self.checkpoint_dir}")
        else:
            self.save_checkpoint(final=True)
            print(f"Training complete. Final checkpoint saved to {self.checkpoint_dir}")

        # Close TensorBoard writer
        self.writer.close()

    def save_checkpoint(self, final=False):
        """Save all model checkpoints."""
        suffix = "final" if final else f"step_{self.train_step}"
        checkpoint = {
            "step": self.train_step,
            "encoder": self.encoder._orig_mod.state_dict(),
            "world_model": {
                k: v for k, v in self.world_model._orig_mod.state_dict().items()
                if k not in ('h_prev', 'z_prev')
            },
            "actor": self.actor._orig_mod.state_dict(),
            "critic": self.critic._orig_mod.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def save_wm_only_checkpoint(self, final=False):
        """Save WM-only checkpoint for bootstrap phase (no actor/critic)."""
        suffix = "final" if final else f"step_{self.train_step}"
        checkpoint = {
            "step": self.train_step,
            "encoder": self.encoder._orig_mod.state_dict(),
            "world_model": {
                k: v for k, v in self.world_model._orig_mod.state_dict().items()
                if k not in ('h_prev', 'z_prev')
            },
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "checkpoint_type": "wm_only",
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
        self.encoder._orig_mod.load_state_dict(checkpoint['encoder'])
        self.world_model._orig_mod.load_state_dict(checkpoint['world_model'], strict=False)

        # Restore WM optimizer if present
        if 'wm_optimizer' in checkpoint:
            self.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])

        # Handle actor/critic based on explicit user intent
        has_ac = 'actor' in checkpoint

        if reset_ac:
            # User explicitly requested fresh actor/critic
            print(f"Loaded WM weights from {checkpoint_path}")
            print("Actor/critic reset to random (--reset-ac)")
            return 'reset_ac'
        elif not has_ac:
            # WM-only checkpoint, no AC to load
            print(f"Loaded WM-only checkpoint from {checkpoint_path}")
            print("Actor/critic initialized randomly")
            return 'wm_only'
        else:
            # Full checkpoint with --resume: load everything
            self.actor._orig_mod.load_state_dict(checkpoint['actor'])
            self.critic._orig_mod.load_state_dict(checkpoint['critic'])

            if 'actor_optimizer' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            if 'critic_optimizer' in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            self.train_step = checkpoint.get('step', 0)
            print(f"Resumed full checkpoint from {checkpoint_path} at step {self.train_step}")
            return 'full'

    def update_actor_critic_losses(
            self,
            dreamed_values_logits,
            dreamed_values,
            lambda_returns,
            dreamed_actions_logits,
            dreamed_actions_sampled
    ):
        
        dreamed_values_logits_flat = dreamed_values_logits.view(-1, dreamed_values_logits.size(-1))
        lambda_returns_flat = lambda_returns.reshape(-1)
        critic_targets = twohot_encode(lambda_returns_flat, self.B)
        
        # Use soft cross-entropy for soft targets (twohot encoding)
        # -sum(targets * log_softmax(logits))
        critic_loss = -torch.sum(critic_targets * F.log_softmax(dreamed_values_logits_flat, dim=-1), dim=-1).mean()

        # Actor Loss: Policy gradient with lambda returns as advantage
        advantage = (lambda_returns - dreamed_values).detach()

        # Normalize advantages for training stability
        if self.normalize_advantages and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        action_dist = torch.distributions.Categorical(logits=dreamed_actions_logits)
        entropy = action_dist.entropy().mean()
        log_probs = action_dist.log_prob(dreamed_actions_sampled)

        # Reinforce algorithm: log_prob * advantage + entropy bonus for exploration
        actor_loss = -torch.mean(log_probs * advantage) - self.actor_entropy_coef * entropy

        return actor_loss, critic_loss, entropy


    def dream_sequence(
        self,
        initial_h,
        initial_z_embed,
        num_dream_steps
    ):
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

            action_dist = torch.distributions.Categorical(logits=action_logits)
            action_sample = action_dist.sample()
            dreamed_actions_sampled.append(action_sample)
            action_onehot = F.one_hot(action_sample, num_classes=self.n_actions).float()

            # 1. Step the dynamics model to get the next h and prior_z
            dream_h_dyn, dream_prior_logits = self.world_model.step_dynamics(
                dream_z_embed, action_onehot, dream_h
            )

            # 2. Sample z from the prior
            dream_prior_dist = dist.Categorical(logits=dream_prior_logits)
            dream_z_sample_indices = dream_prior_dist.sample()
            dream_z_sample = F.one_hot(
                dream_z_sample_indices, num_classes=self.d_hidden // 16
            ).float()

            # 3. Form the full state (h, z) for the next iteration's predictions
            dream_h = self.world_model.join_h_and_z(dream_h_dyn, dream_z_sample).detach()
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
        num_dream_steps
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
        # Observation pixels are bernoulli, while observation vectors are gaussian
        pixel_probs = obs_reconstruction["pixels"]
        obs_pred = symlog(obs_reconstruction["state"])  # Apply symlog only to state vector

        pixel_target = obs_t["pixels"]
        obs_target = obs_t["state"]
        obs_target = symlog(obs_target)  # loss in symlog space
        beta_dyn = config.train.beta_dyn
        beta_rep = config.train.beta_rep
        beta_pred = config.train.beta_pred

        # There are three loss terms:
        # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
        # a. dynamics represetnation
        # -ln p(x|z,h) is trained with symlog squared loss
        pred_loss_vector = 1 / 2 * (obs_pred - obs_target) ** 2
        pred_loss_vector = pred_loss_vector.mean()

        bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
        # The decoder outputs logits, and the target should be in [0,1]
        # Target pixels are already resized to target_size when loading data
        pred_loss_pixel = bce_with_logits_loss_fn(
            input=pixel_probs, target=pixel_target / 255.0
        )

        reward_target = twohot_encode(reward_t, self.B)
        # Use soft cross-entropy for soft targets (twohot encoding)
        # reward_dist should be logits, reward_target is probabilities
        reward_loss = -torch.sum(reward_target * F.log_softmax(reward_dist, dim=-1), dim=-1).mean()

        # c. continue predictor
        # The target is 1 if we continue, 0 if we terminate.
        continue_target = (1.0 - terminated_t.float()).unsqueeze(-1)
        pred_loss_continue = bce_with_logits_loss_fn(
            continue_logits, continue_target
        )

        # Prediction loss is the sum of the individual losses
        l_pred = (
            pred_loss_pixel + pred_loss_vector + reward_loss + pred_loss_continue
        )

        # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
        # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
        # Log-likelihoods. Torch accepts logits

        # The "free bits" technique provides a minimum budget for the KL divergence.
        prior_dist = dist.Categorical(logits=prior_logits)
        free_bits = 1.0
        l_dyn_raw = dist.kl_divergence(
            dist.Categorical(logits=posterior_dist.logits.detach()),
            prior_dist,
        ).mean()
        l_dyn = torch.max(torch.tensor(free_bits, device=self.device), l_dyn_raw)

        l_rep_raw = dist.kl_divergence(
            posterior_dist,
            dist.Categorical(logits=prior_dist.logits.detach()),
        ).mean()
        l_rep = torch.max(torch.tensor(free_bits, device=self.device), l_rep_raw)

        total_loss = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep
        
        # Return both total loss and individual components for logging
        loss_dict = {
            'prediction_pixel': pred_loss_pixel,
            'prediction_vector': pred_loss_vector,
            'prediction_reward': reward_loss,
            'prediction_continue': pred_loss_continue,
            'dynamics': l_dyn,
            'representation': l_rep,
            'kl_dynamics_raw': l_dyn_raw,
            'kl_representation_raw': l_rep_raw,
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
        last_dreamed_pixels=None
    ):
        """Log metrics to TensorBoard."""
        step = self.train_step
        
        # Safety check - should not happen due to assertions, but just in case
        if total_wm_loss is None or total_actor_loss is None or total_critic_loss is None:
            return
        
        # All losses normalized to per-step for fair comparison
        if sequence_length > 0:
            norm = 1.0 / sequence_length
            beta_pred = config.train.beta_pred
            beta_dyn = config.train.beta_dyn
            beta_rep = config.train.beta_rep

            # Per-step totals
            wm_per_step = total_wm_loss.item() * norm
            self.writer.add_scalar("loss/world_model/total_per_step", wm_per_step, step)
            self.writer.add_scalar("loss/actor/total_per_step", total_actor_loss.item() * norm, step)
            self.writer.add_scalar("loss/critic/total_per_step", total_critic_loss.item() * norm, step)

            # Raw component values (per-step, unscaled)
            pixel = wm_loss_components['prediction_pixel'] * norm
            vector = wm_loss_components['prediction_vector'] * norm
            reward = wm_loss_components['prediction_reward'] * norm
            cont = wm_loss_components['prediction_continue'] * norm
            dyn = wm_loss_components['dynamics'] * norm
            rep = wm_loss_components['representation'] * norm

            # Prediction sub-components (unscaled, for debugging)
            self.writer.add_scalar("loss/wm_components/pixel", pixel, step)
            self.writer.add_scalar("loss/wm_components/vector", vector, step)
            self.writer.add_scalar("loss/wm_components/reward", reward, step)
            self.writer.add_scalar("loss/wm_components/continue", cont, step)
            self.writer.add_scalar("loss/wm_components/dynamics", dyn, step)
            self.writer.add_scalar("loss/wm_components/representation", rep, step)

            # Scaled contributions to total (these should sum to total_per_step)
            pred_total = pixel + vector + reward + cont
            self.writer.add_scalar("loss/wm_scaled/prediction", beta_pred * pred_total, step)
            self.writer.add_scalar("loss/wm_scaled/dynamics", beta_dyn * dyn, step)
            self.writer.add_scalar("loss/wm_scaled/representation", beta_rep * rep, step)

            # Raw KL divergences (before free bits clipping)
            self.writer.add_scalar("debug/kl_dynamics_raw", wm_loss_components['kl_dynamics_raw'] * norm, step)
            self.writer.add_scalar("debug/kl_representation_raw", wm_loss_components['kl_representation_raw'] * norm, step)
        else:
            # Fallback if no sequence
            self.writer.add_scalar("loss/world_model/total_per_step", total_wm_loss.item(), step)
            self.writer.add_scalar("loss/actor/total_per_step", total_actor_loss.item(), step)
            self.writer.add_scalar("loss/critic/total_per_step", total_critic_loss.item(), step)
        
        # Dreamed trajectory statistics (debugging metrics)
        if dreamed_rewards_list:
            all_dreamed_rewards = torch.cat(dreamed_rewards_list, dim=0)
            self.writer.add_scalar("debug/dream/reward/mean", all_dreamed_rewards.mean().item(), step)
            self.writer.add_scalar("debug/dream/reward/std", all_dreamed_rewards.std().item(), step)
            self.writer.add_scalar("debug/dream/reward/min", all_dreamed_rewards.min().item(), step)
            self.writer.add_scalar("debug/dream/reward/max", all_dreamed_rewards.max().item(), step)
        
        if dreamed_values_list:
            all_dreamed_values = torch.cat(dreamed_values_list, dim=0)
            self.writer.add_scalar("debug/dream/value/mean", all_dreamed_values.mean().item(), step)
            self.writer.add_scalar("debug/dream/value/std", all_dreamed_values.std().item(), step)
        
        # Actor entropy (important for monitoring exploration)
        if actor_entropy_list:
            all_entropy = torch.stack(actor_entropy_list)  # stack scalars into 1D tensor
            self.writer.add_scalar("actor/entropy/mean", all_entropy.mean().item(), step)
            self.writer.add_scalar("actor/entropy/std", all_entropy.std().item(), step)
        
        # Learning rates
        self.writer.add_scalar("training/learning_rate/world_model", self.wm_optimizer.param_groups[0]['lr'], step)
        self.writer.add_scalar("training/learning_rate/actor", self.actor_optimizer.param_groups[0]['lr'], step)
        self.writer.add_scalar("training/learning_rate/critic", self.critic_optimizer.param_groups[0]['lr'], step)

        # Visualizations every 50 steps (show first sample only)
        if step % 50 == 0 and last_obs_pixels is not None and last_reconstruction_pixels is not None:
            # Take first sample from batch
            actual = (last_obs_pixels[0] / 255.0).clamp(0, 1)  # (C, H, W)
            recon = torch.sigmoid(last_reconstruction_pixels[0]).clamp(0, 1)  # (C, H, W)
            # Resize reconstruction to match actual if needed
            if recon.shape != actual.shape:
                recon = F.interpolate(recon.unsqueeze(0), size=actual.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

            # 1. Actual vs Reconstruction side by side
            comparison = torch.cat([actual, recon], dim=2)  # concat on width
            self.writer.add_image("images/actual_vs_reconstruction", comparison, step)

            # 2. Reconstruction error heatmap (absolute diff, averaged over RGB)
            error = torch.abs(actual - recon).mean(dim=0, keepdim=True)  # (1, H, W)
            error_norm = error / (error.max() + 1e-8)
            error_heatmap = error_norm.repeat(3, 1, 1)  # grayscale to RGB
            self.writer.add_image("images/reconstruction_error", error_heatmap, step)

            # 3. Latent activation heatmap
            if last_posterior_probs is not None:
                # Shape: (batch, d_hidden, categories) -> take first batch, make 2D
                latent_probs = last_posterior_probs[0]  # (512, 32)
                # Normalize and add batch/channel dims for add_images
                latent_img = latent_probs.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 32)
                self.writer.add_images("images/latent_activations", latent_img, step)

            # 4. Dream rollout video (imagined future frames)
            if last_dreamed_pixels is not None:
                # Shape: (n_dream_steps, batch, C, H, W) -> take first batch
                dream_frames = last_dreamed_pixels[:, 0]  # (n_steps, C, H, W)
                # Resize to match actual size if needed
                if dream_frames.shape[2:] != actual.shape[2:]:
                    dream_frames = F.interpolate(dream_frames, size=actual.shape[2:], mode='bilinear', align_corners=False)
                # Add video: shape (N, T, C, H, W) - batch, time, channels, height, width
                video = dream_frames.unsqueeze(0)  # (1, n_steps, C, H, W)
                self.writer.add_video("video/dream_rollout", video, step, fps=4)
                # Also add strip image for quick glance
                n_show = min(5, dream_frames.shape[0])
                dream_strip = torch.cat([dream_frames[i] for i in range(n_show)], dim=2)
                self.writer.add_images("images/dream_rollout", dream_strip.unsqueeze(0), step)

            # 5. Original resolution image (larger, easier to see)
            if last_obs_pixels_original is not None:
                original = (last_obs_pixels_original[0] / 255.0).clamp(0, 1)  # First sample only
                self.writer.add_image("images/original_resolution", original, step)

        self.writer.flush()

    def send_models_to_collector(self, training_step):
        # Use _orig_mod to get state_dict without torch.compile prefix
        # Exclude h_prev/z_prev buffers as they have batch-size-dependent shapes
        wm_state = {
            k: v.cpu() for k, v in self.world_model._orig_mod.state_dict().items()
            if k not in ('h_prev', 'z_prev')
        }
        models_to_send = {
            "actor": {k: v.cpu() for k, v in self.actor._orig_mod.state_dict().items()},
            "encoder": {k: v.cpu() for k, v in self.encoder._orig_mod.state_dict().items()},
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

def train_world_model(config, data_queue, model_queue, log_dir, checkpoint_path=None, mode='train', reset_ac=False):
    trainer = WorldModelTrainer(config, data_queue, model_queue, log_dir, checkpoint_path, mode, reset_ac)
    trainer.train_models()