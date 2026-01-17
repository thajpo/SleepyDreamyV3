"""Dream sequence generation for DreamerV3 imagination-based learning."""

import torch
import torch.nn.functional as F
import torch.distributions as dist


def dream_sequence(
    initial_h,
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
        initial_h: Initial recurrent state (batch, h_dim)
        initial_z_embed: Initial z embedding (batch, z_embed_dim)
        num_dream_steps: Number of imagination steps
        actor: Actor network for action selection
        world_model: World model with step_dynamics and join_h_and_z methods
        n_actions: Number of discrete actions
        d_hidden: Hidden dimension for z sampling

    Returns:
        Tuple of:
            - dreamed_recurrent_states: (num_steps, batch, h_dim)
            - dreamed_actions_logits: (num_steps, batch, n_actions)
            - dreamed_actions_sampled: (num_steps, batch)
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
        action_logits = actor(dream_h.detach())
        dreamed_actions_logits.append(action_logits)

        action_dist = dist.Categorical(logits=action_logits, validate_args=False)
        action_sample = action_dist.sample()
        dreamed_actions_sampled.append(action_sample)
        action_onehot = F.one_hot(action_sample, num_classes=n_actions).float()

        # 1. Step the dynamics model to get the next h and prior_z
        dream_h_dyn, dream_prior_logits = world_model.step_dynamics(
            dream_z_embed, action_onehot, dream_h
        )

        # 2. Sample z from the prior
        dream_prior_dist = dist.Categorical(
            logits=dream_prior_logits, validate_args=False
        )
        dream_z_sample_indices = dream_prior_dist.sample()
        dream_z_sample = F.one_hot(
            dream_z_sample_indices, num_classes=d_hidden // 16
        ).float()

        # 3. Form the full state (h, z) for the next iteration's predictions
        dream_h = world_model.join_h_and_z(
            dream_h_dyn, dream_z_sample
        ).detach()
        dream_z_embed = world_model.z_embedding(
            dream_z_sample.view(dream_z_sample.size(0), -1)
        ).detach()

    # Stack the collected dreamed states and actions
    return (
        torch.stack(dreamed_recurrent_states),
        torch.stack(dreamed_actions_logits),
        torch.stack(dreamed_actions_sampled),
    )


def calculate_lambda_returns(
    dreamed_rewards,
    dreamed_values,
    dreamed_continues,
    gamma,
    lam,
    num_dream_steps,
):
    """
    Calculate lambda-returns (GAE-style) for a dreamed trajectory.

    Lambda-returns provide a balance between bias and variance in value estimation
    by mixing n-step returns with different horizons.

    Args:
        dreamed_rewards: (num_steps, batch) predicted rewards
        dreamed_values: (num_steps, batch) predicted values
        dreamed_continues: (num_steps, batch) continue logits
        gamma: Discount factor
        lam: Lambda parameter for GAE (0=TD(0), 1=Monte Carlo)
        num_dream_steps: Number of dream steps

    Returns:
        Lambda returns tensor of shape (num_steps, batch)
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
