"""Loss computation functions for DreamerV3 world model training."""

import torch
import torch.nn.functional as F

from .math_utils import symlog, twohot_encode, unimix_logits


def compute_wm_loss(
    obs_reconstruction,
    obs_t,
    reward_dist,
    reward_t,
    terminated_t,
    continue_logits,
    posterior_dist,
    prior_logits,
    B,
    config,
    device,
    use_pixels=True,
    sample_mask=None,
):
    """
    Compute world model loss combining prediction, dynamics, and representation losses.

    Args:
        obs_reconstruction: Dict with "state" and optionally "pixels" reconstructions
        obs_t: Dict with "state" and optionally "pixels" targets
        reward_dist: Reward distribution logits from world model
        reward_t: Target rewards
        terminated_t: Termination flags
        continue_logits: Continue predictor logits
        posterior_dist: Posterior distribution from encoder
        prior_logits: Prior logits from dynamics model
        B: Bin tensor for twohot encoding
        config: Config object with beta coefficients
        device: Torch device
        use_pixels: Whether to compute pixel loss

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual components
    """
    beta_dyn = config.beta_dyn
    beta_rep = config.beta_rep
    beta_pred = config.beta_pred

    # There are three loss terms:
    # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
    # a. dynamics representation
    # -ln p(x|z,h) is trained with symlog squared loss
    # State-vector reconstruction loss is disabled for pixel-only envs (n_observations=0).
    state_pred = obs_reconstruction.get("state")
    state_target = obs_t.get("state")
    has_state_targets = (
        state_pred is not None
        and state_target is not None
        and state_pred.ndim >= 2
        and state_target.ndim >= 2
        and state_pred.shape[-1] > 0
        and state_target.shape[-1] > 0
    )
    if has_state_targets:
        obs_pred = symlog(state_pred)
        obs_target = symlog(state_target)  # loss in symlog space
        pred_loss_vector = 1 / 2 * (obs_pred - obs_target) ** 2
        pred_loss_vector = pred_loss_vector.mean(dim=-1)  # (B,)
    else:
        pred_loss_vector = torch.zeros(
            reward_t.shape[0], device=device, dtype=reward_t.dtype
        )

    # Pixel loss (only when using pixels)
    if use_pixels and "pixels" in obs_reconstruction and "pixels" in obs_t:
        pixel_probs = obs_reconstruction["pixels"]
        pixel_target = obs_t["pixels"]
        pixel_bce = F.binary_cross_entropy_with_logits(
            pixel_probs, pixel_target / 255.0, reduction="none"
        )
        pred_loss_pixel = pixel_bce.mean(dim=(1, 2, 3))  # (B,)
    else:
        pred_loss_pixel = torch.zeros(
            pred_loss_vector.shape[0], device=device, dtype=pred_loss_vector.dtype
        )

    reward_target = twohot_encode(reward_t, B)
    # Use soft cross-entropy for soft targets (twohot encoding)
    # reward_dist should be logits, reward_target is probabilities
    reward_loss = -torch.sum(
        reward_target * F.log_softmax(reward_dist, dim=-1), dim=-1
    )  # (B,)

    # c. continue predictor
    # The target is 1 if we continue, 0 if we terminate.
    continue_target = (1.0 - terminated_t.float()).unsqueeze(-1)
    pred_loss_continue = F.binary_cross_entropy_with_logits(
        continue_logits, continue_target, reduction="none"
    ).squeeze(-1)  # (B,)

    # Prediction loss is the sum of the individual losses
    l_pred = pred_loss_pixel + pred_loss_vector + reward_loss + pred_loss_continue

    # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
    # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
    # Log-likelihoods. Torch accepts logits

    # The "free bits" technique provides a minimum budget for the KL divergence.
    free_bits = 1.0
    # Manual categorical KL to avoid distribution overhead.
    prior_logits_mixed = unimix_logits(prior_logits, unimix_ratio=0.01)
    posterior_logits_detached = posterior_dist.logits.detach()
    log_posterior_detached = F.log_softmax(posterior_logits_detached, dim=-1)
    log_prior = F.log_softmax(prior_logits_mixed, dim=-1)
    posterior_probs_detached = log_posterior_detached.exp()
    l_dyn_raw = (
        (posterior_probs_detached * (log_posterior_detached - log_prior))
        .sum(dim=-1)
        .mean(dim=-1)
    )  # (B,)

    log_posterior = F.log_softmax(posterior_dist.logits, dim=-1)
    log_prior_detached = F.log_softmax(prior_logits_mixed.detach(), dim=-1)
    posterior_probs = log_posterior.exp()
    l_rep_raw = (
        (posterior_probs * (log_posterior - log_prior_detached))
        .sum(dim=-1)
        .mean(dim=-1)
    )  # (B,)

    # Straight-through estimator for free bits:
    # Forward: loss = max(free_bits, raw) for reporting/scaling
    # Backward: gradient flows through raw value (not killed by max)
    l_dyn = l_dyn_raw + (free_bits - l_dyn_raw).clamp(min=0).detach()
    l_rep = l_rep_raw + (free_bits - l_rep_raw).clamp(min=0).detach()

    total_loss_per_sample = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

    if sample_mask is None:
        total_loss = total_loss_per_sample.mean()
        mask = None
    else:
        mask = sample_mask.float()
        total_loss = (total_loss_per_sample * mask).sum() / (mask.sum() + 1e-8)

    def _masked_mean(x):
        if mask is None:
            return x.mean()
        return (x * mask).sum() / (mask.sum() + 1e-8)

    # Return both total loss and individual components for logging
    loss_dict = {
        "prediction_pixel": _masked_mean(pred_loss_pixel),
        "prediction_vector": _masked_mean(pred_loss_vector),
        "prediction_reward": _masked_mean(reward_loss),
        "prediction_continue": _masked_mean(pred_loss_continue),
        "dynamics": _masked_mean(l_dyn),
        "representation": _masked_mean(l_rep),
        "kl_dynamics_raw": _masked_mean(l_dyn_raw),
        "kl_representation_raw": _masked_mean(l_rep_raw),
    }

    return total_loss, loss_dict


def compute_actor_critic_losses(
    dreamed_values_logits,
    dreamed_values,
    lambda_returns,
    dreamed_actions_logits,
    dreamed_actions_sampled,
    B,
    S,  # EMA for returns
    actor_entropy_coef=0.003,
    dreamed_values_logits_ema=None,
    critic_ema_coef=1.0,
    sample_mask=None,
):
    """
    Compute actor and critic losses for policy gradient training.

    Args:
        dreamed_values_logits: Value function logits from critic
        dreamed_values: Decoded value estimates
        lambda_returns: Computed lambda returns
        dreamed_actions_logits: Action logits from actor
        dreamed_actions_sampled: Sampled actions
        B: Bin tensor for twohot encoding
        S: EMA for returns
        actor_entropy_coef: Entropy regularization coefficient
        dreamed_values_logits_ema: Logits from EMA critic (for regularization)
        critic_ema_coef: Coefficient for EMA regularization

    Returns:
        Tuple of (actor_loss, critic_loss, entropy)
    """
    num_bins = dreamed_values_logits.size(-1)
    H, Bsz = lambda_returns.shape[:2]
    dreamed_values_logits_flat = dreamed_values_logits.view(-1, num_bins)
    # Detach lambda_returns: critic targets should not have gradients flowing back
    # through dreamed_values (which is part of lambda_returns). This matches
    # DreamerV3's use of sg() (stop_gradient) on value targets.
    lambda_returns_flat = lambda_returns.detach().reshape(-1)
    critic_targets = twohot_encode(lambda_returns_flat, B)

    # Use soft cross-entropy for soft targets (twohot encoding)
    # -sum(targets * log_softmax(logits))
    per_step_ce = -torch.sum(
        critic_targets * F.log_softmax(dreamed_values_logits_flat, dim=-1), dim=-1
    ).view(H, Bsz)

    if sample_mask is not None:
        mask = sample_mask.float().view(1, Bsz)
        critic_loss = (per_step_ce * mask).sum() / (mask.sum() * H + 1e-8)
    else:
        critic_loss = per_step_ce.mean()

    # --- Critic EMA Regularizer (Distributional) ---
    if dreamed_values_logits_ema is not None:
        ema_logits_flat = dreamed_values_logits_ema.view(-1, num_bins).detach()

        # Target distribution from EMA critic
        ema_probs = F.softmax(ema_logits_flat, dim=-1)

        # Cross-entropy: -sum(P_ema * log(P_current))
        # Note: P_ema is fixed target (detached), P_current has gradients
        per_step_ema = -torch.sum(
            ema_probs * F.log_softmax(dreamed_values_logits_flat, dim=-1), dim=-1
        ).view(H, Bsz)
        if sample_mask is not None:
            mask = sample_mask.float().view(1, Bsz)
            ema_reg_loss = (per_step_ema * mask).sum() / (mask.sum() * H + 1e-8)
        else:
            ema_reg_loss = per_step_ema.mean()

        critic_loss += critic_ema_coef * ema_reg_loss
    # -----------------------------------------------

    # Actor Loss: Policy gradient with lambda returns as advantage
    # advantage = (lambda_returns - dreamed_values).detach()

    # Normalize advantages for training stability
    # if normalize_advantages and advantage.numel() > 1:
    # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    advantage = ((lambda_returns - dreamed_values) / max(1, S)).detach()

    action_dist = torch.distributions.Categorical(
        logits=dreamed_actions_logits, validate_args=False
    )
    entropy = action_dist.entropy()  # (H, B)
    log_probs = action_dist.log_prob(dreamed_actions_sampled)  # (H, B)

    # Reinforce algorithm: log_prob * advantage + entropy bonus for exploration
    per_step_actor = -(log_probs * advantage) - actor_entropy_coef * entropy
    if sample_mask is not None:
        mask = sample_mask.float().view(1, Bsz)
        actor_loss = (per_step_actor * mask).sum() / (mask.sum() * H + 1e-8)
        entropy = (entropy * mask).sum() / (mask.sum() * H + 1e-8)
    else:
        actor_loss = per_step_actor.mean()
        entropy = entropy.mean()

    return actor_loss, critic_loss, entropy
