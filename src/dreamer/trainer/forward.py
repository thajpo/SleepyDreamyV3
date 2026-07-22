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
    compute_reinforce_actor_loss,
    compute_q_actor_critic_losses,
    compute_enumerated_actor_loss,
    compute_mpc_teacher_actor_loss,
    dream_sequence,
    calculate_lambda_returns,
    learned_continue_discount,
    symlog,
    symexp,
    twohot_encode,
    twohot_expectation,
    unimix_logits,
)
from .logging import StepMetrics, collect_viz_data


@dataclass
class ForwardResult:
    """Everything the training loop needs from one forward pass."""

    total_wm_loss: torch.Tensor
    total_actor_loss: torch.Tensor
    total_critic_loss: torch.Tensor
    metrics: StepMetrics
    replay_representation_loss: Optional[torch.Tensor] = None
    return_scale: Optional[float] = None
    ret_lo: Optional[float] = None
    ret_hi: Optional[float] = None


@dataclass
class ReinforceActorBatch:
    """Tensors needed to apply one shared return scale to an imagined batch."""

    dreamed_values: torch.Tensor
    lambda_returns: torch.Tensor
    dreamed_continues: torch.Tensor
    dreamed_actions_logits: torch.Tensor
    dreamed_actions_sampled: torch.Tensor
    sample_mask: torch.Tensor
    actor_baseline_values: torch.Tensor


def calculate_return_normalizer_update(
    return_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ret_lo: float,
    ret_hi: float,
    *,
    rate: float = 0.01,
) -> Optional[tuple[float, float, float]]:
    """Update DreamerV3's non-debiased percentile return normalizer.

    ``return_batches`` contains ``(lambda_returns, start_mask)`` pairs for every
    post-burn-in imagination start. Reference Dreamer computes percentiles over
    the corresponding combined ``B * K * H`` tensor, not only the final start.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("return-normalizer rate must be between 0 and 1")

    valid_returns: list[torch.Tensor] = []
    for lambda_returns, start_mask in return_batches:
        valid = start_mask.detach().bool().reshape(-1)
        if bool(valid.any().item()):
            valid_returns.append(lambda_returns.detach()[:, valid].reshape(-1))
    if not valid_returns:
        return None

    flat = torch.cat(valid_returns)
    lo_batch = float(torch.quantile(flat, 0.05).item())
    hi_batch = float(torch.quantile(flat, 0.95).item())
    next_lo = (1.0 - rate) * float(ret_lo) + rate * lo_batch
    next_hi = (1.0 - rate) * float(ret_hi) + rate * hi_batch
    next_scale = max(1.0, next_hi - next_lo)
    return next_scale, next_lo, next_hi


def add_sequence_mean_auxiliary_loss(
    accumulated_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    scale: float,
    sequence_length: int,
) -> torch.Tensor:
    """Add one averaged auxiliary loss beside a summed sequence loss.

    The caller later divides ``accumulated_loss`` by ``sequence_length``.
    Multiplying the already-averaged auxiliary term here preserves its authored
    scale after that normalization instead of diluting it by the sequence
    length.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return accumulated_loss + sequence_length * scale * auxiliary_loss


def calculate_replay_lambda_targets(
    rewards: torch.Tensor,
    is_last: torch.Tensor,
    is_terminal: torch.Tensor,
    value_annotations: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Return value targets aligned with post-transition replay states.

    Replay row ``t`` stores the observation produced by the action and reward
    in row ``t``. Its posterior therefore represents the state *after* that
    reward, so its value target must start with the next row's reward. The last
    posterior has no following replay transition and intentionally has no
    target here; its imagined value annotation bootstraps the preceding state.
    """
    num_targets = max(0, rewards.shape[0] - 1)
    if num_targets == 0:
        return rewards[:0]
    next_rewards = rewards[1:]
    next_last = is_last[1:].float()
    next_terminal = is_terminal[1:].float()
    next_bootstrap = value_annotations[1:]

    live = float(gamma) * (1.0 - next_terminal)
    trace = float(lam) * (1.0 - next_last)
    next_return = value_annotations[-1]
    returns: list[torch.Tensor] = []
    for index in reversed(range(num_targets)):
        next_return = (
            next_rewards[index]
            + (1.0 - trace[index]) * live[index] * next_bootstrap[index]
            + live[index] * trace[index] * next_return
        )
        returns.append(next_return)
    return torch.stack(list(reversed(returns)))


def calculate_replay_pair_mask(
    replay_mask: torch.Tensor, replay_is_last: torch.Tensor
) -> torch.Tensor:
    """Mask invalid and cross-episode posterior/target pairs."""
    return (
        replay_mask[:-1]
        * replay_mask[1:]
        * (1.0 - replay_is_last[:-1].float())
    )


def select_critic_target_values(
    online_values: torch.Tensor,
    slow_values: torch.Tensor,
    *,
    use_slow_target: bool,
) -> torch.Tensor:
    """Select a detached value target for returns and the policy baseline.

    Reference DreamerV3 uses the online value prediction for these targets by
    default and retains the slow value model as a distributional regularizer.
    The explicit switch preserves the semantics of historical local runs that
    instead used the slow model for both roles.
    """
    target = slow_values if use_slow_target else online_values
    return target.detach()


def dreamer_step(
    *,
    encoder,
    world_model,
    actor,
    critic,
    critic_ema,
    q_critic=None,
    q_critic_ema=None,
    batch,
    metrics: StepMetrics,
    all_tokens: torch.Tensor,
    B: int,
    T: int,
    train_start_t: int,
    skip_actor: bool,
    skip_critic: bool,
    bins: torch.Tensor,
    return_scale: float,
    config,
    device: torch.device,
    use_pixels: bool,
    do_log_images: bool,
    collect_gradient_diagnostics: bool = False,
    return_lo: float = 0.0,
    return_hi: float = 0.0,
    return_norm_rate: float = 0.01,
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
        critic_ema: Slow critic used for distributional regularization and,
            when configured, historical lambda-return/baseline targets.
        batch: EnvData named tuple from replay buffer.
        metrics: Pre-initialized StepMetrics accumulator.
        all_tokens: Encoder output, shape (B, T, token_dim).
        B: Batch size.
        T: Sequence length.
        train_start_t: First timestep for loss accumulation (after burn-in).
        skip_actor: Whether to skip actor updates this batch.
        skip_critic: Whether to skip critic updates this batch.
        bins: Symexp bin edges for distributional value/reward.
        config: Training config.
        device: Torch device.
        use_pixels: Whether pixel observations are active.
        do_log_images: Whether to collect visualization data this step.
        collect_gradient_diagnostics: Whether to measure read-only replay/WM
            gradient alignment for this step.

    Returns:
        ForwardResult with accumulated losses and populated metrics.
    """
    effective_train_steps = T - train_start_t
    n_actions = config.n_actions
    n_dream_steps = config.num_dream_steps
    d_hidden = config.d_hidden
    gamma = config.gamma
    imagination_discount = learned_continue_discount(
        gamma, bool(getattr(config, "contdisc", True))
    )
    lam = config.lam
    actor_entropy_coef = config.actor_entropy_coef
    critic_ema_coef = config.critic_ema_regularizer
    critic_replay_scale = config.critic_replay_scale
    critic_real_return_scale = float(getattr(config, "critic_real_return_scale", 0.0))
    q_critic_scale = float(getattr(config, "q_critic_scale", 0.0))
    prior_state_pred_scale = float(getattr(config, "prior_state_pred_scale", 0.0))

    total_wm_loss = None
    total_actor_loss = None
    total_critic_loss = None
    return_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    reinforce_actor_batches: list[ReinforceActorBatch] = []
    replay_posterior_states_with_grad: list[torch.Tensor] = []
    replay_representation_loss = None

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
            continue_loss_weights=batch.continue_weights[:, t_step],
        )
        valid_rows = sample_mask.bool()
        terminal_rows = valid_rows & is_terminal_t.bool()
        live_rows = valid_rows & ~is_terminal_t.bool()
        continue_probs = torch.sigmoid(continue_logits.detach().squeeze(-1))
        if bool(terminal_rows.any().item()):
            metrics.continue_terminal_weights.append(
                batch.continue_weights[:, t_step][terminal_rows].detach()
            )
            metrics.continue_terminal_probs.append(continue_probs[terminal_rows])
        if bool(live_rows.any().item()):
            metrics.continue_live_probs.append(continue_probs[live_rows])
        if (
            prior_state_pred_scale > 0.0
            and not use_pixels
            and obs_t.get("state") is not None
            and obs_t["state"].shape[-1] > 0
        ):
            h_dim = config.d_hidden * config.rnn_n_blocks
            h_state = h_z_joined[:, :h_dim]
            prior_probs = F.softmax(unimix_logits(prior_logits, unimix_ratio=0.01), dim=-1)
            prior_h_z = world_model.join_h_and_z(h_state, prior_probs)
            prior_state_pred = world_model.decoder(prior_h_z).get("state")
            if prior_state_pred is not None:
                per_sample_prior = 0.5 * (
                    prior_state_pred - symlog(obs_t["state"])
                ) ** 2
                per_sample_prior = per_sample_prior.mean(dim=-1)
                prior_mask = sample_mask.float()
                prior_state_loss = (per_sample_prior * prior_mask).sum() / (
                    prior_mask.sum() + 1e-8
                )
                wm_loss = wm_loss + prior_state_pred_scale * prior_state_loss
                wm_loss_dict["prior_state"] = prior_state_loss.detach()

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
        state_prediction = obs_reconstruction.get("state")
        state_target = obs_t.get("state")
        if (
            state_prediction is not None
            and state_target is not None
            and state_prediction.shape[-1] > 0
        ):
            metrics.replay_states.append(state_target.detach())
            metrics.replay_state_reconstructions.append(
                symexp(state_prediction.detach())
            )
            metrics.replay_state_masks.append(sample_mask.detach())

        if critic_replay_scale > 0.0 or critic_real_return_scale > 0.0:
            metrics.replay_posterior_states.append(h_z_joined.detach())
        if collect_gradient_diagnostics and critic_replay_scale > 0.0:
            replay_posterior_states_with_grad.append(h_z_joined)

        for key in metrics.wm_components:
            metrics.wm_components[key] = (
                metrics.wm_components[key] + wm_loss_dict[key].detach()
            )

        # --- Dream sequence for actor-critic ---
        valid_ac_step = sample_mask.sum() > 0
        if (not skip_actor or not skip_critic) and valid_ac_step:
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
            dreamed_rewards = twohot_expectation(
                dreamed_rewards_logits, bins
            ).detach()
            dreamed_continues = (
                world_model.continue_predictor(dreamed_recurrent_states[1:])
                .detach()
                .squeeze(-1)
            )
            terminal_reward_penalty = float(
                getattr(config, "terminal_reward_penalty", 0.0)
            )
            if terminal_reward_penalty:
                continue_probs = torch.sigmoid(dreamed_continues)
                dreamed_rewards = dreamed_rewards - terminal_reward_penalty * (
                    1.0 - continue_probs
                )
            metrics.dreamed_rewards.append(dreamed_rewards.detach().cpu())

            dreamed_values_logits = critic(dreamed_recurrent_states)
            with torch.no_grad():
                dreamed_values_logits_ema = critic_ema(dreamed_recurrent_states)
                dreamed_values_ema = twohot_expectation(
                    dreamed_values_logits_ema, bins
                )
            dreamed_values = twohot_expectation(dreamed_values_logits, bins)
            metrics.dreamed_values.append(dreamed_values.detach().cpu())

            critic_target_values = select_critic_target_values(
                dreamed_values,
                dreamed_values_ema,
                use_slow_target=bool(getattr(config, "critic_slow_target", True)),
            )

            lambda_returns = calculate_lambda_returns(
                dreamed_rewards,
                critic_target_values,
                dreamed_continues,
                imagination_discount,
                lam,
                n_dream_steps,
            )
            return_batches.append((lambda_returns, sample_mask))

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
                imagination_discount,
                actor_entropy_coef=actor_entropy_coef,
                normalize_advantages=config.normalize_advantages,
                dreamed_values_logits_ema=dreamed_values_logits_ema,
                critic_ema_coef=critic_ema_coef,
                sample_mask=sample_mask,
                actor_baseline_values=critic_target_values,
            )
            if (
                not skip_actor
                and getattr(config, "actor_loss_mode", "reinforce") == "reinforce"
            ):
                reinforce_actor_batches.append(
                    ReinforceActorBatch(
                        dreamed_values=dreamed_values,
                        lambda_returns=lambda_returns,
                        dreamed_continues=dreamed_continues,
                        dreamed_actions_logits=dreamed_actions_logits,
                        dreamed_actions_sampled=dreamed_actions_sampled,
                        sample_mask=sample_mask,
                        actor_baseline_values=critic_target_values,
                    )
                )
            if (
                not skip_actor
                and getattr(config, "actor_loss_mode", "reinforce") == "enumerate"
            ):
                (
                    actor_loss,
                    entropy,
                    _q_values,
                    enum_margin,
                ) = compute_enumerated_actor_loss(
                    h_z_joined,
                    world_model.z_embedding(posterior_z_sample.view(B, -1)),
                    actor,
                    critic_ema,
                    world_model,
                    n_actions,
                    d_hidden,
                    bins,
                    imagination_discount,
                    getattr(config, "actor_enum_horizon", 3),
                    actor_entropy_coef,
                    getattr(config, "actor_enum_temperature", 0.25),
                    terminal_reward_penalty=terminal_reward_penalty,
                    objective=getattr(config, "actor_enum_objective", "value"),
                    sample_mask=sample_mask,
                )
                actor_loss = actor_loss * float(
                    getattr(config, "actor_enum_loss_scale", 1.0)
                )
                metrics.actor_enum_margin.append(enum_margin.detach().cpu())
            if (
                not skip_actor
                and getattr(config, "actor_loss_mode", "reinforce") == "mpc_teacher"
            ):
                (
                    actor_loss,
                    entropy,
                    _q_values,
                    mpc_margin,
                    mpc_mask_frac,
                ) = compute_mpc_teacher_actor_loss(
                    h_z_joined,
                    world_model.z_embedding(posterior_z_sample.view(B, -1)),
                    actor,
                    critic_ema,
                    world_model,
                    n_actions,
                    d_hidden,
                    bins,
                    imagination_discount,
                    getattr(config, "mpc_teacher_horizon", 6),
                    actor_entropy_coef,
                    getattr(config, "mpc_teacher_temperature", 0.1),
                    terminal_reward_penalty=terminal_reward_penalty,
                    objective=getattr(config, "mpc_teacher_objective", "value"),
                    target=getattr(config, "mpc_teacher_target", "hard"),
                    margin_min=getattr(config, "mpc_teacher_margin_min", 0.0),
                    normalize_values=getattr(
                        config, "mpc_teacher_normalize_values", True
                    ),
                    sample_mask=sample_mask,
                )
                actor_loss = actor_loss * float(
                    getattr(config, "mpc_teacher_loss_scale", 1.0)
                )
                metrics.actor_mpc_margin.append(mpc_margin.detach().cpu())
                metrics.actor_mpc_mask_frac.append(mpc_mask_frac.detach().cpu())
            if (
                q_critic is not None
                and q_critic_ema is not None
                and (
                    (not skip_critic and q_critic_scale > 0.0)
                    or (
                        not skip_actor
                        and getattr(config, "actor_loss_mode", "reinforce")
                        == "qcritic"
                    )
                )
            ):
                num_bins = bins.numel()
                q_logits = q_critic(dreamed_recurrent_states[:-1])
                q_logits = q_logits.view(
                    n_dream_steps, B, n_actions, num_bins
                )
                with torch.no_grad():
                    q_logits_ema = q_critic_ema(dreamed_recurrent_states)
                    q_logits_ema = q_logits_ema.view(
                        n_dream_steps + 1, B, n_actions, num_bins
                    )
                    q_values_ema = twohot_expectation(q_logits_ema, bins)
                    q_state_values = q_values_ema.max(dim=-1).values
                    q_lambda_returns = calculate_lambda_returns(
                        dreamed_rewards,
                        q_state_values,
                        dreamed_continues,
                        imagination_discount,
                        lam,
                        n_dream_steps,
                    )

                (
                    q_actor_loss,
                    q_loss,
                    q_entropy,
                    q_margin,
                ) = compute_q_actor_critic_losses(
                    q_logits,
                    q_values_ema[:-1],
                    q_lambda_returns,
                    dreamed_continues,
                    dreamed_actions_logits,
                    dreamed_actions_sampled,
                    bins,
                    imagination_discount,
                    actor_entropy_coef=actor_entropy_coef,
                    temperature=getattr(config, "q_actor_temperature", 0.25),
                    sample_mask=sample_mask,
                )
                if not skip_critic:
                    critic_loss = critic_loss + q_critic_scale * q_loss
                if (
                    not skip_actor
                    and getattr(config, "actor_loss_mode", "reinforce") == "qcritic"
                ):
                    actor_loss = q_actor_loss
                    entropy = q_entropy
                metrics.actor_q_margin.append(q_margin.detach().cpu())
            metrics.actor_entropy.append(entropy.detach().cpu())
            if skip_actor:
                actor_loss = torch.tensor(0.0, device=device)
            if skip_critic:
                critic_loss = torch.tensor(0.0, device=device)
        else:
            actor_loss = torch.tensor(0.0, device=device)
            critic_loss = torch.tensor(0.0, device=device)
            if critic_replay_scale > 0.0 and not skip_critic:
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

    normalizer_update = calculate_return_normalizer_update(
        return_batches,
        return_lo,
        return_hi,
        rate=return_norm_rate,
    )
    next_return_scale = return_scale
    next_ret_lo = return_lo
    next_ret_hi = return_hi
    if normalizer_update is not None:
        next_return_scale, next_ret_lo, next_ret_hi = normalizer_update

        # The reference updates its return normalizer from all imagination
        # starts before evaluating the current policy loss. Critic losses do
        # not depend on this scale, so only the default REINFORCE actor loss
        # needs to be recomputed after the combined percentile is known.
        if reinforce_actor_batches:
            recomputed_actor_losses = []
            for actor_batch in reinforce_actor_batches:
                recomputed_actor_loss, _entropy = compute_reinforce_actor_loss(
                    actor_batch.dreamed_values,
                    actor_batch.lambda_returns,
                    actor_batch.dreamed_continues,
                    actor_batch.dreamed_actions_logits,
                    actor_batch.dreamed_actions_sampled,
                    next_return_scale,
                    imagination_discount,
                    actor_entropy_coef=actor_entropy_coef,
                    normalize_advantages=config.normalize_advantages,
                    sample_mask=actor_batch.sample_mask,
                    actor_baseline_values=actor_batch.actor_baseline_values,
                )
                recomputed_actor_losses.append(recomputed_actor_loss)
            total_actor_loss = torch.stack(recomputed_actor_losses).sum()

    # --- Replay critic grounding ---
    if (
        critic_replay_scale > 0.0
        and not skip_critic
        and effective_train_steps > 1
        and len(metrics.replay_value_annotations) == effective_train_steps
        and len(metrics.replay_posterior_states) == effective_train_steps
    ):
        replay_posterior = torch.stack(metrics.replay_posterior_states, dim=1)
        replay_annotations = torch.stack(metrics.replay_value_annotations, dim=0)

        replay_rewards = batch.rewards[:, train_start_t:].transpose(0, 1)
        replay_is_last = batch.is_last[:, train_start_t:].transpose(0, 1)
        replay_is_terminal = batch.is_terminal[:, train_start_t:].transpose(0, 1)
        replay_mask = batch.mask[:, train_start_t:].transpose(0, 1)

        replay_lambda_returns = calculate_replay_lambda_targets(
            replay_rewards,
            replay_is_last,
            replay_is_terminal,
            replay_annotations,
            gamma,
            lam,
        ).transpose(0, 1)

        replay_logits = critic(replay_posterior[:, :-1].detach())
        logits_flat = replay_logits.reshape(-1, replay_logits.size(-1))
        targets_flat = replay_lambda_returns.detach().reshape(-1)
        replay_pair_mask = calculate_replay_pair_mask(replay_mask, replay_is_last)
        mask_flat = replay_pair_mask.transpose(0, 1).reshape(-1).float()

        targets_twohot = twohot_encode(targets_flat, bins)
        per_step_ce = -torch.sum(
            targets_twohot * F.log_softmax(logits_flat, dim=-1), dim=-1
        )
        metrics.replay_loss = (per_step_ce * mask_flat).sum() / (
            mask_flat.sum() + 1e-8
        )

        with torch.no_grad():
            replay_logits_ema = critic_ema(replay_posterior[:, :-1].detach())
        ema_logits_flat = replay_logits_ema.reshape(-1, replay_logits_ema.size(-1))
        ema_probs = F.softmax(ema_logits_flat, dim=-1)
        per_step_ema = -torch.sum(
            ema_probs * F.log_softmax(logits_flat, dim=-1), dim=-1
        )
        metrics.replay_ema_reg = (per_step_ema * mask_flat).sum() / (
            mask_flat.sum() + 1e-8
        )

        replay_loss_total = (
            metrics.replay_loss + config.critic_ema_regularizer * metrics.replay_ema_reg
        )
        total_critic_loss = add_sequence_mean_auxiliary_loss(
            total_critic_loss,
            replay_loss_total,
            critic_replay_scale,
            effective_train_steps,
        )

        if (
            collect_gradient_diagnostics
            and len(replay_posterior_states_with_grad) == effective_train_steps
        ):
            live_replay_posterior = torch.stack(
                replay_posterior_states_with_grad, dim=1
            )
            live_replay_logits = critic(live_replay_posterior[:, :-1])
            live_logits_flat = live_replay_logits.reshape(
                -1, live_replay_logits.size(-1)
            )
            live_per_step_ce = -torch.sum(
                targets_twohot * F.log_softmax(live_logits_flat, dim=-1),
                dim=-1,
            )
            live_target_loss = (live_per_step_ce * mask_flat).sum() / (
                mask_flat.sum() + 1e-8
            )
            live_per_step_ema = -torch.sum(
                ema_probs * F.log_softmax(live_logits_flat, dim=-1),
                dim=-1,
            )
            live_slow_regularizer = (
                live_per_step_ema * mask_flat
            ).sum() / (mask_flat.sum() + 1e-8)
            replay_representation_loss = (
                effective_train_steps
                * critic_replay_scale
                * (
                    live_target_loss
                    + config.critic_ema_regularizer * live_slow_regularizer
                )
            )

    # --- Full-episode replay return critic grounding ---
    if (
        critic_real_return_scale > 0.0
        and not skip_critic
        and len(metrics.replay_posterior_states) == effective_train_steps
    ):
        if batch.future_returns is None:
            raise RuntimeError(
                "critic_real_return_scale requires replay future-return annotations"
            )
        replay_posterior = torch.stack(metrics.replay_posterior_states, dim=1)
        replay_logits = critic(replay_posterior.detach())
        logits_flat = replay_logits.reshape(-1, replay_logits.size(-1))
        targets_flat = (
            batch.future_returns[:, train_start_t:].detach().reshape(-1)
        )
        mask_flat = batch.mask[:, train_start_t:].reshape(-1).float()
        targets_twohot = twohot_encode(targets_flat, bins)
        per_step_ce = -torch.sum(
            targets_twohot * F.log_softmax(logits_flat, dim=-1), dim=-1
        )
        metrics.replay_mc_loss = (per_step_ce * mask_flat).sum() / (
            mask_flat.sum() + 1e-8
        )
        total_critic_loss = add_sequence_mean_auxiliary_loss(
            total_critic_loss,
            metrics.replay_mc_loss,
            critic_real_return_scale,
            effective_train_steps,
        )

    assert total_wm_loss is not None and total_actor_loss is not None and total_critic_loss is not None

    return ForwardResult(
        total_wm_loss=total_wm_loss,
        total_actor_loss=total_actor_loss,
        total_critic_loss=total_critic_loss,
        metrics=metrics,
        replay_representation_loss=replay_representation_loss,
        return_scale=next_return_scale,
        ret_lo=next_ret_lo,
        ret_hi=next_ret_hi,
    )
