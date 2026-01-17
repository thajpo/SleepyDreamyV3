"""Loss computation functions for DreamerV3 world model training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .math_utils import symlog, twohot_encode


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
    # Observation vectors use symlog squared loss
    obs_pred = symlog(obs_reconstruction["state"])
    obs_target = symlog(obs_t["state"])  # loss in symlog space
    beta_dyn = config.train.beta_dyn
    beta_rep = config.train.beta_rep
    beta_pred = config.train.beta_pred

    # There are three loss terms:
    # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
    # a. dynamics representation
    # -ln p(x|z,h) is trained with symlog squared loss
    pred_loss_vector = 1 / 2 * (obs_pred - obs_target) ** 2
    pred_loss_vector = pred_loss_vector.mean()

    # Pixel loss (only when using pixels)
    if use_pixels and "pixels" in obs_reconstruction and "pixels" in obs_t:
        pixel_probs = obs_reconstruction["pixels"]
        pixel_target = obs_t["pixels"]
        bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
        pred_loss_pixel = bce_with_logits_loss_fn(
            input=pixel_probs, target=pixel_target / 255.0
        )
    else:
        pred_loss_pixel = torch.tensor(0.0, device=device)

    reward_target = twohot_encode(reward_t, B)
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
    free_bits = 1.0
    # Manual categorical KL to avoid distribution overhead.
    posterior_logits_detached = posterior_dist.logits.detach()
    log_posterior_detached = F.log_softmax(posterior_logits_detached, dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    posterior_probs_detached = log_posterior_detached.exp()
    l_dyn_raw = (
        (posterior_probs_detached * (log_posterior_detached - log_prior))
        .sum(dim=-1)
        .mean()
    )

    log_posterior = F.log_softmax(posterior_dist.logits, dim=-1)
    log_prior_detached = F.log_softmax(prior_logits.detach(), dim=-1)
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


def compute_actor_critic_losses(
    dreamed_values_logits,
    dreamed_values,
    lambda_returns,
    dreamed_actions_logits,
    dreamed_actions_sampled,
    B,
    S,  # EMA for returns
    actor_entropy_coef=0.003,
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
        normalize_advantages: Whether to normalize advantages
        actor_entropy_coef: Entropy regularization coefficient

    Returns:
        Tuple of (actor_loss, critic_loss, entropy)
    """
    dreamed_values_logits_flat = dreamed_values_logits.view(
        -1, dreamed_values_logits.size(-1)
    )
    lambda_returns_flat = lambda_returns.reshape(-1)
    critic_targets = twohot_encode(lambda_returns_flat, B)

    # Use soft cross-entropy for soft targets (twohot encoding)
    # -sum(targets * log_softmax(logits))
    critic_loss = -torch.sum(
        critic_targets * F.log_softmax(dreamed_values_logits_flat, dim=-1), dim=-1
    ).mean()

    # Actor Loss: Policy gradient with lambda returns as advantage
    # advantage = (lambda_returns - dreamed_values).detach()

    # Normalize advantages for training stability
    # if normalize_advantages and advantage.numel() > 1:
    # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    advantage = ((lambda_returns - dreamed_values) / max(1, S)).detach()

    action_dist = torch.distributions.Categorical(
        logits=dreamed_actions_logits, validate_args=False
    )
    entropy = action_dist.entropy().mean()
    log_probs = action_dist.log_prob(dreamed_actions_sampled)

    # Reinforce algorithm: log_prob * advantage + entropy bonus for exploration
    actor_loss = -torch.mean(log_probs * advantage) - actor_entropy_coef * entropy

    return actor_loss, critic_loss, entropy
