"""Dream sequence generation for DreamerV3 imagination-based learning."""

import torch
import torch.nn.functional as F
import torch.distributions as dist

from .math_utils import unimix_logits


def dream_sequence(
    initial_h_z,
    initial_z_embed,
    num_dream_steps,
    actor,
    world_model,
    n_actions,
    d_hidden,
):
    """
    Generate a sequence of dreamed states and actions starting from an initial state.

    The world model rollout is treated as a fixed simulator for actor/critic updates.
    Gradients do not flow back into the world model when optimizing the policy or value head.

    Args:
        initial_h_z: Initial joined state (batch, h_dim + z_dim) where h_dim = n_blocks * d_hidden
        initial_z_embed: Initial z embedding (batch, z_embed_dim)
        num_dream_steps: Number of imagination steps
        actor: Actor network for action selection
        world_model: World model with step_dynamics and join_h_and_z methods
        n_actions: Number of discrete actions
        d_hidden: Hidden dimension for z sampling

    Returns:
        Tuple of:
            - dreamed_recurrent_states: (num_steps + 1, batch, h_z_dim)
              where index 0 is the initial state and index t+1 is post-action state
            - dreamed_actions_logits: (num_steps, batch, n_actions)
            - dreamed_actions_sampled: (num_steps, batch)
    """
    dreamed_recurrent_states = []
    dreamed_actions_logits = []
    dreamed_actions_sampled = []

    # Treat the world model rollout as a fixed simulator for actor/critic updates.
    # We detach the starting state and all subsequent transitions so gradients do
    # not flow back into the world model when optimizing the policy or value head.
    #
    # Track two state representations:
    # - dream_h_z: joined (h, z) state for actor input (h_dim + z_dim)
    # - dream_h_state: just h for step_dynamics (h_dim = n_blocks * d_hidden)
    h_dim = world_model.n_blocks * d_hidden
    dream_h_z = initial_h_z.detach()
    dream_h_state = dream_h_z[:, :h_dim]  # Extract h from joined state
    dream_z_embed = initial_z_embed.detach()

    # Include the starting state so values align with actions at the same index.
    dreamed_recurrent_states.append(dream_h_z)

    for _ in range(num_dream_steps):
        action_logits = actor(dream_h_z.detach())
        action_logits = unimix_logits(
            action_logits, unimix_ratio=0.01
        )  # Actor unimix (1%)
        dreamed_actions_logits.append(action_logits)

        action_dist = dist.Categorical(logits=action_logits, validate_args=False)
        action_sample = action_dist.sample()
        dreamed_actions_sampled.append(action_sample)
        action_onehot = F.one_hot(action_sample, num_classes=n_actions).float()

        # 1. Step the dynamics model to get the next h and prior_z
        dream_h_dyn, dream_prior_logits = world_model.step_dynamics(
            dream_z_embed, action_onehot, dream_h_state
        )

        # 2. Sample z from the prior with unimix (DreamerV3 Section 4)
        dream_prior_logits_mixed = unimix_logits(dream_prior_logits, unimix_ratio=0.01)
        dream_prior_dist = dist.Categorical(
            logits=dream_prior_logits_mixed, validate_args=False
        )
        dream_z_sample_indices = dream_prior_dist.sample()
        dream_z_sample = F.one_hot(
            dream_z_sample_indices, num_classes=world_model.n_classes
        ).float()

        # 3. Update both state representations for next iteration.
        # Record post-transition state so reward/value heads align with sampled action.
        dream_h_state = dream_h_dyn.detach()
        dream_h_z = world_model.join_h_and_z(dream_h_dyn, dream_z_sample).detach()
        dreamed_recurrent_states.append(dream_h_z)
        dream_z_embed = world_model.z_embedding(
            dream_z_sample.view(dream_z_sample.size(0), -1)
        ).detach()

    # Stack the collected dreamed states and actions
    return (
        torch.stack(dreamed_recurrent_states),
        torch.stack(dreamed_actions_logits),
        torch.stack(dreamed_actions_sampled),
    )


@torch.no_grad()
def enumerate_first_action_values(
    initial_h_z,
    initial_z_embed,
    actor,
    critic,
    world_model,
    n_actions,
    d_hidden,
    bins,
    gamma,
    horizon,
    terminal_reward_penalty=0.0,
    objective="value",
    bootstrap_value=True,
):
    """Estimate Q(s, a) by enumerating discrete first actions in the world model.

    This is a low-variance control diagnostic for small discrete action spaces:
    instead of sampling one action and assigning it a REINFORCE target, it asks
    the learned model what would happen after each possible first action from
    the same latent state. Future actions are enumerated as a tiny deterministic
    tree and the best branch for each first action is used as its value. Set
    ``bootstrap_value`` to false to isolate learned reward and continuation
    rollout values from the critic's terminal-state estimate.
    """
    horizon = max(1, int(horizon))
    batch_size = initial_h_z.shape[0]
    h_dim = world_model.n_blocks * d_hidden
    device = initial_h_z.device
    dtype = initial_h_z.dtype

    h_state = initial_h_z[:, :h_dim].detach().unsqueeze(1)
    z_embed = initial_z_embed.detach().unsqueeze(1)
    z_state = None
    first_action = None
    returns = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    discounts = torch.ones(batch_size, 1, device=device, dtype=dtype)
    branch_count = 1

    action_template = torch.arange(n_actions, device=device)

    for depth in range(horizon):
        new_branch_count = branch_count * n_actions
        action_ids = (
            action_template.view(1, 1, n_actions)
            .expand(batch_size, branch_count, n_actions)
            .reshape(batch_size, new_branch_count)
        )
        action_onehot = F.one_hot(action_ids.reshape(-1), num_classes=n_actions).to(
            dtype=dtype
        )

        h_flat = (
            h_state.unsqueeze(2)
            .expand(batch_size, branch_count, n_actions, h_dim)
            .reshape(batch_size * new_branch_count, h_dim)
        )
        z_embed_flat = (
            z_embed.unsqueeze(2)
            .expand(batch_size, branch_count, n_actions, z_embed.shape[-1])
            .reshape(batch_size * new_branch_count, z_embed.shape[-1])
        )

        h_next, prior_logits = world_model.step_dynamics(
            z_embed_flat, action_onehot, h_flat
        )
        prior_logits = unimix_logits(prior_logits, unimix_ratio=0.01)
        z_state_flat = F.softmax(prior_logits, dim=-1)
        h_z_flat = world_model.join_h_and_z(h_next, z_state_flat)

        continue_probs = torch.sigmoid(
            world_model.continue_predictor(h_z_flat).squeeze(-1)
        )
        if objective == "survival":
            rewards = continue_probs
        else:
            reward_logits = world_model.reward_predictor(h_z_flat)
            rewards = torch.sum(F.softmax(reward_logits, dim=-1) * bins, dim=-1)
            if terminal_reward_penalty:
                rewards = rewards - float(terminal_reward_penalty) * (
                    1.0 - continue_probs
                )

        returns = (
            returns.unsqueeze(2)
            .expand(batch_size, branch_count, n_actions)
            .reshape(batch_size, new_branch_count)
        )
        discounts = (
            discounts.unsqueeze(2)
            .expand(batch_size, branch_count, n_actions)
            .reshape(batch_size, new_branch_count)
        )
        returns = returns + discounts * rewards.view(batch_size, new_branch_count)
        discounts = discounts * gamma * continue_probs.view(
            batch_size, new_branch_count
        )

        if depth == 0:
            first_action = action_ids
        else:
            first_action = (
                first_action.unsqueeze(2)
                .expand(batch_size, branch_count, n_actions)
                .reshape(batch_size, new_branch_count)
            )

        h_state = h_next.view(batch_size, new_branch_count, h_dim).detach()
        z_state = z_state_flat.view(
            batch_size, new_branch_count, world_model.n_latents, world_model.n_classes
        ).detach()
        z_embed = world_model.z_embedding(
            z_state.reshape(batch_size * new_branch_count, -1)
        ).view(batch_size, new_branch_count, -1)
        branch_count = new_branch_count

    assert first_action is not None and z_state is not None
    final_h_z = world_model.join_h_and_z(
        h_state.reshape(batch_size * branch_count, h_dim),
        z_state.reshape(
            batch_size * branch_count, world_model.n_latents, world_model.n_classes
        ),
    )
    if objective == "survival" or not bootstrap_value:
        values = torch.zeros(batch_size, branch_count, device=device, dtype=dtype)
    else:
        value_logits = critic(final_h_z)
        values = torch.sum(F.softmax(value_logits, dim=-1) * bins, dim=-1).view(
            batch_size, branch_count
        )
    path_values = returns + discounts * values

    q_values = []
    neg_inf = torch.tensor(-torch.inf, device=device, dtype=dtype)
    for action in range(n_actions):
        action_values = path_values.masked_fill(first_action != action, neg_inf)
        q_values.append(action_values.max(dim=1).values)
    return torch.stack(q_values, dim=-1)


def compute_enumerated_actor_loss(
    initial_h_z,
    initial_z_embed,
    actor,
    critic,
    world_model,
    n_actions,
    d_hidden,
    bins,
    gamma,
    horizon,
    actor_entropy_coef,
    temperature,
    terminal_reward_penalty=0.0,
    objective="value",
    sample_mask=None,
):
    """Train the actor toward model-enumerated action values for discrete control."""
    h_prev_backup = world_model.h_prev.clone()
    try:
        q_values = enumerate_first_action_values(
            initial_h_z,
            initial_z_embed,
            actor,
            critic,
            world_model,
            n_actions,
            d_hidden,
            bins,
            gamma,
            horizon,
            terminal_reward_penalty=terminal_reward_penalty,
            objective=objective,
        )
    finally:
        world_model.h_prev = h_prev_backup
    policy_logits = unimix_logits(actor(initial_h_z.detach()), unimix_ratio=0.01)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    target_probs = F.softmax(q_values / max(float(temperature), 1e-6), dim=-1)
    action_dist = dist.Categorical(logits=policy_logits, validate_args=False)
    entropy_per_sample = action_dist.entropy()
    per_sample_loss = -torch.sum(target_probs * log_probs, dim=-1)
    per_sample_loss = per_sample_loss - actor_entropy_coef * entropy_per_sample

    if sample_mask is not None:
        mask = sample_mask.float()
        loss = (per_sample_loss * mask).sum() / (mask.sum() + 1e-8)
        entropy = (entropy_per_sample * mask).sum() / (mask.sum() + 1e-8)
        margin = (
            (q_values.topk(k=min(2, n_actions), dim=-1).values[:, 0]
             - q_values.topk(k=min(2, n_actions), dim=-1).values[:, -1])
            * mask
        ).sum() / (mask.sum() + 1e-8)
    else:
        loss = per_sample_loss.mean()
        entropy = entropy_per_sample.mean()
        q_top = q_values.topk(k=min(2, n_actions), dim=-1).values
        margin = (q_top[:, 0] - q_top[:, -1]).mean()

    return loss, entropy, q_values.detach(), margin.detach()


def compute_mpc_teacher_actor_loss(
    initial_h_z,
    initial_z_embed,
    actor,
    critic,
    world_model,
    n_actions,
    d_hidden,
    bins,
    gamma,
    horizon,
    actor_entropy_coef,
    temperature,
    terminal_reward_penalty=0.0,
    objective="value",
    target="hard",
    margin_min=0.0,
    normalize_values=True,
    sample_mask=None,
):
    """Distill a short-horizon planner teacher into the actor.

    This is deliberately stricter than the older enumerate loss. The model plans
    all short action sequences, then the actor is supervised toward the best
    first action only on states where the teacher has enough margin. That avoids
    pushing the policy on near-ties where the learned model is likely noisy.
    """
    h_prev_backup = world_model.h_prev.clone()
    try:
        q_values = enumerate_first_action_values(
            initial_h_z,
            initial_z_embed,
            actor,
            critic,
            world_model,
            n_actions,
            d_hidden,
            bins,
            gamma,
            horizon,
            terminal_reward_penalty=terminal_reward_penalty,
            objective=objective,
        )
    finally:
        world_model.h_prev = h_prev_backup

    policy_logits = unimix_logits(actor(initial_h_z.detach()), unimix_ratio=0.01)
    action_dist = dist.Categorical(logits=policy_logits, validate_args=False)
    entropy_per_sample = action_dist.entropy()

    q_for_target = q_values
    if normalize_values and q_for_target.shape[-1] > 1:
        q_mean = q_for_target.mean(dim=-1, keepdim=True)
        q_std = q_for_target.std(dim=-1, keepdim=True).clamp_min(1e-6)
        q_for_target = (q_for_target - q_mean) / q_std

    top = q_values.topk(k=min(2, n_actions), dim=-1).values
    if n_actions > 1:
        margins = top[:, 0] - top[:, 1]
    else:
        margins = torch.zeros(q_values.shape[0], device=q_values.device)

    target_actions = q_values.argmax(dim=-1)
    if target == "soft":
        log_probs = F.log_softmax(policy_logits, dim=-1)
        target_probs = F.softmax(q_for_target / max(float(temperature), 1e-6), dim=-1)
        per_sample_loss = -torch.sum(target_probs * log_probs, dim=-1)
    elif target == "hard":
        per_sample_loss = F.cross_entropy(
            policy_logits, target_actions, reduction="none"
        )
    else:
        raise ValueError(f"Unsupported mpc teacher target: {target}")

    per_sample_loss = per_sample_loss - actor_entropy_coef * entropy_per_sample

    teacher_mask = margins >= float(margin_min)
    if sample_mask is not None:
        teacher_mask = teacher_mask & sample_mask.bool()
    mask = teacher_mask.float()
    mask_denom = mask.sum() + 1e-8
    loss = (per_sample_loss * mask).sum() / mask_denom
    entropy = (entropy_per_sample * mask).sum() / mask_denom
    margin = (margins * mask).sum() / mask_denom
    mask_fraction = mask.mean()

    return loss, entropy, q_values.detach(), margin.detach(), mask_fraction.detach()


def calculate_lambda_returns(
    dreamed_rewards,
    dreamed_values,
    dreamed_continues,
    gamma,
    lam,
    num_dream_steps,
    value_annotations=None,
    continues_are_logits=True,
):
    """
    Calculate lambda-returns (GAE-style) for a dreamed trajectory.

    Lambda-returns provide a balance between bias and variance in value estimation
    by mixing n-step returns with different horizons.

    Args:
        dreamed_rewards: (num_steps, batch) predicted rewards
        dreamed_values: (num_steps, batch) predicted values
        dreamed_continues: (num_steps, batch) continue logits or probabilities
        gamma: Discount factor
        lam: Lambda parameter for GAE (0=TD(0), 1=Monte Carlo)
        num_dream_steps: Number of dream steps
        value_annotations: Optional (num_steps, batch) value annotations. If provided,
            these replace dreamed_values in the (1-λ) term and in the bootstrap.
        continues_are_logits: If True, applies sigmoid to dreamed_continues.

    Returns:
        Lambda returns tensor of shape (num_steps, batch)
    """
    lambda_returns = []
    if value_annotations is None:
        value_source = dreamed_values
    else:
        value_source = value_annotations

    next_lambda_return = value_source[-1]

    # Iterate backwards through the trajectory.
    # TD(lambda) uses next-step value in the (1-lambda) branch:
    # G_t = r_t + gamma*c_t*((1-lambda)V_{t+1} + lambda*G_{t+1})
    for i in reversed(range(num_dream_steps)):
        reward_t = dreamed_rewards[i]
        if continues_are_logits:
            continue_prob_t = torch.sigmoid(dreamed_continues[i])
        else:
            continue_prob_t = dreamed_continues[i]

        if i + 1 < num_dream_steps:
            next_value = value_source[i + 1]
        else:
            next_value = value_source[-1]

        next_lambda_return = reward_t + gamma * continue_prob_t * (
            (1 - lam) * next_value + lam * next_lambda_return
        )
        lambda_returns.append(next_lambda_return)

    # The returns are calculated backwards, so reverse to [t=0..H-1].
    return torch.stack(lambda_returns).flip(0)
