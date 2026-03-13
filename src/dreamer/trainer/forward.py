"""DreamerV3 forward pass: RSSM rollout + dreaming + loss computation.

This is the core algorithm, extracted from the training loop so it can
be read and tested independently.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..models import (
    compute_wm_loss,
    compute_actor_critic_losses,
    dream_sequence,
    calculate_lambda_returns,
    twohot_encode,
)
from .logging import StepMetrics, collect_viz_data


@dataclass
class ForwardResult:
    """Everything the training loop needs from one forward pass."""

    total_wm_loss: torch.Tensor
    total_actor_loss: torch.Tensor
    total_critic_loss: torch.Tensor
    metrics: StepMetrics
    # Return lambda_returns from the last AC step so the caller can
    # update return-normalization EMA without reaching into internals.
    last_lambda_returns: Optional[torch.Tensor] = None


def dreamer_step(
    *,
    encoder,
    world_model,
    actor,
    critic,
    critic_ema,
    batch,
    metrics: StepMetrics,
    all_tokens: torch.Tensor,
    B: int,
    T: int,
    train_start_t: int,
    skip_ac: bool,
    bins: torch.Tensor,
    return_scale: float,
    config,
    device: torch.device,
    use_pixels: bool,
    do_log_images: bool,
) -> ForwardResult:
    """One complete DreamerV3 forward pass over a batch.

    Runs the RSSM rollout over T timesteps, dreams for actor-critic on
    post-burn-in steps, computes replay critic grounding, and returns
    accumulated losses + metrics.

    Args:
        encoder: Observation encoder (already used to produce all_tokens).
        world_model: RSSM world model (h_prev/z_prev already zeroed).
        actor: Policy network.
        critic: Value network.
        critic_ema: EMA copy of critic for stable targets.
        batch: EnvData named tuple from replay buffer.
        metrics: Pre-initialized StepMetrics accumulator.
        all_tokens: Encoder output, shape (B, T, token_dim).
        B: Batch size.
        T: Sequence length.
        train_start_t: First timestep for loss accumulation (after burn-in).
        skip_ac: Whether to skip actor-critic updates this batch.
        bins: Symexp bin edges for distributional value/reward.
        config: Training config.
        device: Torch device.
        use_pixels: Whether pixel observations are active.
        do_log_images: Whether to collect visualization data this step.

    Returns:
        ForwardResult with accumulated losses and populated metrics.
    """
    effective_train_steps = T - train_start_t
    n_actions = config.n_actions
    n_dream_steps = config.num_dream_steps
    d_hidden = config.d_hidden
    gamma = config.gamma
    lam = config.lam
    actor_entropy_coef = config.actor_entropy_coef
    critic_ema_coef = config.critic_ema_regularizer
    critic_replay_scale = config.critic_replay_scale

    total_wm_loss = None
    total_actor_loss = None
    total_critic_loss = None
    last_lambda_returns = None

    # --- RSSM rollout + per-timestep dreaming ---
    for t_step in range(T):
        if use_pixels and batch.pixels is not None:
            obs_t = {
                "pixels": batch.pixels[:, t_step],
                "state": batch.states[:, t_step],
            }
        else:
            obs_t = {"state": batch.states[:, t_step]}
        action_t = batch.actions[:, t_step]
        reward_t = batch.rewards[:, t_step]
        is_terminal_t = batch.is_terminal[:, t_step]
        sample_mask = batch.mask[:, t_step]
        tokens_t = all_tokens[:, t_step]

        (
            obs_reconstruction,
            reward_dist,
            continue_logits,
            h_z_joined,
            posterior_z_sample,
            prior_logits,
            posterior_logits,
        ) = world_model(tokens_t, action_t)

        # World model loss
        wm_loss, wm_loss_dict = compute_wm_loss(
            obs_reconstruction,
            obs_t,
            reward_dist,
            reward_t,
            is_terminal_t,
            continue_logits,
            posterior_logits,
            prior_logits,
            bins,
            config,
            device,
            use_pixels=use_pixels,
            sample_mask=sample_mask,
        )

        collect_viz_data(
            metrics, t_step, T, obs_t, obs_reconstruction,
            posterior_logits, batch, use_pixels,
        )

        # Default zero AC losses for burn-in timesteps
        actor_loss = torch.tensor(0.0, device=device)
        critic_loss = torch.tensor(0.0, device=device)

        if t_step < train_start_t:
            # Burn-in: only accumulate WM loss, skip AC
            if total_wm_loss is None:
                total_wm_loss = wm_loss
                total_actor_loss = actor_loss
                total_critic_loss = critic_loss
            else:
                total_wm_loss = total_wm_loss + wm_loss
            continue

        # --- Post-burn-in: accumulate WM components for logging ---
        if critic_replay_scale > 0.0:
            metrics.replay_posterior_states.append(h_z_joined.detach())

        for key in metrics.wm_components:
            metrics.wm_components[key] = (
                metrics.wm_components[key] + wm_loss_dict[key].detach()
            )

        # --- Dream sequence for actor-critic ---
        valid_ac_step = sample_mask.sum() > 0
        if not skip_ac and valid_ac_step:
            h_prev_backup = world_model.h_prev.clone()
            (
                dreamed_recurrent_states,
                dreamed_actions_logits,
                dreamed_actions_sampled,
            ) = dream_sequence(
                h_z_joined,
                world_model.z_embedding(posterior_z_sample.view(B, -1)),
                n_dream_steps,
                actor,
                world_model,
                n_actions,
                d_hidden,
            )
            world_model.h_prev = h_prev_backup

            # Reward/continue predictions from post-action states [1:]
            dreamed_rewards_logits = world_model.reward_predictor(
                dreamed_recurrent_states[1:]
            ).detach()
            dreamed_rewards_probs = F.softmax(dreamed_rewards_logits, dim=-1)
            dreamed_rewards = torch.sum(
                dreamed_rewards_probs * bins, dim=-1
            ).detach()
            metrics.dreamed_rewards.append(dreamed_rewards.detach().cpu())

            dreamed_continues = (
                world_model.continue_predictor(dreamed_recurrent_states[1:])
                .detach()
                .squeeze(-1)
            )

            dreamed_values_logits = critic(dreamed_recurrent_states)
            with torch.no_grad():
                dreamed_values_logits_ema = critic_ema(dreamed_recurrent_states)
            dreamed_values_probs = F.softmax(dreamed_values_logits, dim=-1)
            dreamed_values = torch.sum(dreamed_values_probs * bins, dim=-1)
            metrics.dreamed_values.append(dreamed_values.detach().cpu())

            lambda_returns = calculate_lambda_returns(
                dreamed_rewards,
                dreamed_values,
                dreamed_continues,
                gamma,
                lam,
                n_dream_steps,
            )
            last_lambda_returns = lambda_returns

            if critic_replay_scale > 0.0:
                metrics.replay_value_annotations.append(lambda_returns[0].detach())

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
                bins,
                return_scale,
                gamma,
                actor_entropy_coef=actor_entropy_coef,
                dreamed_values_logits_ema=dreamed_values_logits_ema,
                critic_ema_coef=critic_ema_coef,
                sample_mask=sample_mask,
            )
            metrics.actor_entropy.append(entropy.detach().cpu())
        else:
            actor_loss = torch.tensor(0.0, device=device)
            critic_loss = torch.tensor(0.0, device=device)
            if critic_replay_scale > 0.0 and not skip_ac:
                metrics.replay_value_annotations.append(
                    torch.zeros_like(sample_mask, device=device)
                )

            # Decode dreamed states for visualization
            if do_log_images and t_step == T - 1 and use_pixels:
                with torch.no_grad():
                    try:
                        n_steps, batch_sz = dreamed_recurrent_states.shape[:2]
                        flat = dreamed_recurrent_states.view(n_steps * batch_sz, -1)
                        dreamed_obs = world_model.decoder(flat)
                        pixels_flat = dreamed_obs.get("pixels")
                        if pixels_flat is not None:
                            C, H, W = pixels_flat.shape[1:]
                            metrics.viz_data["dreamed_pixels"] = torch.sigmoid(
                                pixels_flat.view(n_steps, batch_sz, C, H, W)
                            ).detach()
                    except Exception:
                        pass

        # Accumulate losses
        if total_wm_loss is None:
            total_wm_loss = wm_loss
            total_actor_loss = actor_loss
            total_critic_loss = critic_loss
        else:
            total_wm_loss = total_wm_loss + wm_loss
            total_actor_loss = total_actor_loss + actor_loss
            total_critic_loss = total_critic_loss + critic_loss

    # --- Replay critic grounding ---
    if (
        critic_replay_scale > 0.0
        and not skip_ac
        and len(metrics.replay_value_annotations) == effective_train_steps
        and len(metrics.replay_posterior_states) == effective_train_steps
    ):
        replay_posterior = torch.stack(metrics.replay_posterior_states, dim=1)
        replay_annotations = torch.stack(metrics.replay_value_annotations, dim=0)

        replay_rewards = batch.rewards[:, train_start_t:].transpose(0, 1)
        replay_is_last = batch.is_last[:, train_start_t:].transpose(0, 1)
        replay_continues = (1.0 - batch.is_terminal.float()).transpose(0, 1)[
            train_start_t:
        ]
        replay_continues = replay_continues * (1.0 - replay_is_last.float())
        replay_continues = replay_continues * (
            1.0 - 1.0 / float(max(1, getattr(config, "horizon", 333)))
        )
        replay_mask = batch.mask[:, train_start_t:].transpose(0, 1)

        replay_lambda_returns = calculate_lambda_returns(
            replay_rewards,
            replay_rewards,
            replay_continues,
            gamma,
            lam,
            effective_train_steps,
            value_annotations=replay_annotations,
            continues_are_logits=False,
        ).transpose(0, 1)

        replay_logits = critic(replay_posterior.detach())
        logits_flat = replay_logits.reshape(-1, replay_logits.size(-1))
        targets_flat = replay_lambda_returns.detach().reshape(-1)
        mask_flat = replay_mask.reshape(-1).float()

        targets_twohot = twohot_encode(targets_flat, bins)
        per_step_ce = -torch.sum(
            targets_twohot * F.log_softmax(logits_flat, dim=-1), dim=-1
        )
        replay_not_last = (1.0 - replay_is_last.float()).reshape(-1)
        replay_weight = mask_flat * replay_not_last
        metrics.replay_loss = (per_step_ce * replay_weight).sum() / (
            replay_weight.sum() + 1e-8
        )

        with torch.no_grad():
            replay_logits_ema = critic_ema(replay_posterior.detach())
        ema_logits_flat = replay_logits_ema.reshape(-1, replay_logits_ema.size(-1))
        ema_probs = F.softmax(ema_logits_flat, dim=-1)
        per_step_ema = -torch.sum(
            ema_probs * F.log_softmax(logits_flat, dim=-1), dim=-1
        )
        metrics.replay_ema_reg = (per_step_ema * replay_weight).sum() / (
            replay_weight.sum() + 1e-8
        )

        replay_loss_total = (
            metrics.replay_loss + config.critic_ema_regularizer * metrics.replay_ema_reg
        )
        total_critic_loss = total_critic_loss + critic_replay_scale * replay_loss_total

    assert total_wm_loss is not None and total_actor_loss is not None and total_critic_loss is not None

    return ForwardResult(
        total_wm_loss=total_wm_loss,
        total_actor_loss=total_actor_loss,
        total_critic_loss=total_critic_loss,
        metrics=metrics,
        last_lambda_returns=last_lambda_returns,
    )
