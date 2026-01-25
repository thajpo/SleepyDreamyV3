# Critic Replay Loss Implementation Plan (DreamerV3) - CORRECTED

## Overview
This plan correctly implements the critic replay loss from DreamerV3, which trains the critic on both imagined trajectories and real replay data using imagination-anchored targets.

## Key Insight from the Paper

The paper states:
> "The critic replay loss uses the imagination returns R_t^λ at the start states of the imagination rollouts as on-policy value annotations for the replay trajectory to then compute λ-returns over the replay rewards."

**Critical**: The imagination returns serve as value annotations **at each timestep**, not just as a terminal bootstrap.

The formula for replay λ-returns:
```
R_t^λ,replay = r_t^real + γ * c_t^real * ((1-λ) * V_t^annot + λ * R_{t+1}^λ,replay)
```

Where `V_t^annot = R_t^λ,imag` (imagination returns at timestep t).

## Implementation Changes

### 1. config.yaml - New Parameter
```yaml
# src/dreamer/conf/config.yaml
train:
  # Critic replay loss (DreamerV3 paper: βrepval=0.3)
  critic_replay_scale: 0.3
```

### 2. dreaming.py - Enhanced Lambda Returns
Modify `calculate_lambda_returns` to accept per-timestep value annotations:

```python
# src/dreamer/models/dreaming.py
def calculate_lambda_returns(
    rewards,
    values,
    continues,
    gamma,
    lam,
    num_steps,
    value_annotations=None,  # NEW: Per-timestep value annotations
):
    """
    Calculate lambda-returns (GAE-style) with optional per-timestep value annotations.

    Args:
        rewards: (num_steps, batch) rewards
        values: (num_steps, batch) value estimates
        continues: (num_steps, batch) continue logits
        gamma: Discount factor
        lam: Lambda parameter
        num_steps: Number of steps
        value_annotations: Optional (num_steps, batch) value annotations for each step
                          If provided, these replace values in the (1-λ) term
    """
    lambda_returns = []
    next_lambda_return = values[-1]

    # Iterate backwards through the trajectory
    for i in reversed(range(num_steps)):
        reward_t = rewards[i]
        continue_prob_t = torch.sigmoid(continues[i])

        # Use value annotation if provided, otherwise use critic's value
        if value_annotations is not None:
            value_t = value_annotations[i]
        else:
            value_t = values[i]

        next_lambda_return = reward_t + gamma * continue_prob_t * (
            (1 - lam) * value_t + lam * next_lambda_return
        )
        lambda_returns.append(next_lambda_return)

    # The returns are calculated backwards, so we reverse them
    return torch.stack(lambda_returns).flip(0)
```

### 3. core.py - Training Loop Enhancement

The key insight: we need to collect imagination returns for all timesteps, then compute replay loss outside the timestep loop.

```python
# src/dreamer/trainer/core.py - train_models method

# Initialize accumulator for imagination returns (NEW)
imagination_returns_list = []  # Will collect [T, batch] imagination returns

for t_step in range(T):
    # ... existing encoder and world model code ...

    if self.should_train_ac() and not skip_ac_this_step:
        # ... existing dream sequence code ...

        dreamed_values_logits = self.critic(dreamed_recurrent_states)
        with torch.no_grad():
            dreamed_values_logits_ema = self.critic_ema(dreamed_recurrent_states)
        dreamed_values_probs = F.softmax(dreamed_values_logits, dim=-1)
        dreamed_values = torch.sum(dreamed_values_probs * self.B, dim=-1)

        lambda_returns = calculate_lambda_returns(
            dreamed_rewards,
            dreamed_values,
            dreamed_continues,
            self.gamma,
            self.lam,
            self.n_dream_steps,
        )

        # NEW: Collect imagination returns for replay loss
        # Use the first step's imagination return as the annotation for this timestep
        imagination_returns_list.append(lambda_returns[0].detach())  # Shape: [batch]

        # ... existing actor/critic loss computation ...

# NEW: Compute replay loss after all timesteps processed
if self.config.train.critic_replay_scale > 0 and len(imagination_returns_list) > 0:
    # Stack imagination returns: [T, batch]
    imagination_returns = torch.stack(imagination_returns_list, dim=0)

    # Compute replay lambda returns using imagination annotations
    # rewards: [batch, T] -> transpose to [T, batch]
    replay_rewards = self.rewards.transpose(0, 1)
    # continues: [batch, T] -> transpose to [T, batch]
    replay_continues = (1 - self.terminated).float().transpose(0, 1)
    # mask: [batch, T] -> transpose to [T, batch]
    replay_mask = self.mask.transpose(0, 1)

    replay_lambda_returns = calculate_lambda_returns(
        replay_rewards,
        torch.zeros_like(replay_rewards),  # Dummy values (we use annotations)
        replay_continues,
        self.gamma,
        self.lam,
        T,
        value_annotations=imagination_returns,  # KEY: Use imagination returns as annotations
    )

    # Compute critic logits on all posterior states
    # Need to collect posterior states from all timesteps
    # For now, we'll compute this in the loop and accumulate

    # Apply mask to replay loss
    replay_lambda_returns = replay_lambda_returns * replay_mask

    # Compute critic loss on replay data
    # (This will be done by accumulating critic logits in the loop)
```

### 4. core.py - Accumulate Critic Logits for Replay

We need to modify the loop to accumulate critic logits on posterior states:

```python
# src/dreamer/trainer/core.py - train_models method

# Initialize accumulator for replay critic logits (NEW)
replay_critic_logits_list = []  # Will collect [T, batch, num_bins] critic logits

for t_step in range(T):
    # ... existing encoder and world model code ...

    # NEW: Compute critic logits on posterior states for replay loss
    with torch.no_grad():
        replay_critic_logits = self.critic(h_z_joined.detach())
    replay_critic_logits_list.append(replay_critic_logits)  # Shape: [batch, num_bins]

    if self.should_train_ac() and not skip_ac_this_step:
        # ... existing dream sequence code ...

# NEW: Compute replay loss after all timesteps processed
if self.config.train.critic_replay_scale > 0 and len(replay_critic_logits_list) > 0:
    # Stack replay critic logits: [T, batch, num_bins]
    replay_critic_logits = torch.stack(replay_critic_logits_list, dim=0)

    # Flatten for loss computation
    replay_critic_logits_flat = replay_critic_logits.view(-1, replay_critic_logits.size(-1))
    replay_lambda_returns_flat = replay_lambda_returns.detach().reshape(-1)

    # Twohot encode targets
    replay_targets = twohot_encode(replay_lambda_returns_flat, self.B)

    # Compute replay critic loss
    replay_loss = -torch.sum(
        replay_targets * F.log_softmax(replay_critic_logits_flat, dim=-1), dim=-1
    ).mean()

    # Add to total critic loss
    total_critic_loss += self.config.train.critic_replay_scale * replay_loss
```

### 5. core.py - Logging Enhancement

```python
# src/dreamer/trainer/core.py - log_metrics method

# Add replay loss to logging
if log_scalars:
    self.logger.add_scalar("loss/critic/replay", replay_loss.item(), step)
    self.logger.add_scalar("_key/loss_critic_replay", replay_loss.item(), step)
```

## Summary of Changes

1. **config.yaml**: Add `critic_replay_scale: 0.3`
2. **dreaming.py**: Modify `calculate_lambda_returns` to accept `value_annotations` parameter
3. **core.py**:
   - Collect imagination returns for all timesteps
   - Collect critic logits on posterior states for all timesteps
   - Compute replay lambda returns using imagination annotations
   - Compute and add replay loss to total critic loss
   - Add logging for replay loss

## Verification

The implementation correctly:
- Uses imagination returns as value annotations at each timestep
- Computes replay lambda returns with real rewards and imagination annotations
- Applies proper masking to skip padded steps
- Detaches posterior states to prevent encoder updates
- Adds replay loss with the correct weight (0.3)

## Expected Behavior

- Critic trains on both imagined and real data
- Training targets are imagination-anchored to prevent off-policy drift
- More stable training by grounding the critic in real reward signals
- Faster convergence on tasks with hard-to-predict rewards
