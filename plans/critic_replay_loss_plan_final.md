# Critic Replay Loss Implementation Plan (DreamerV3) - FINAL

## Core Concept

DreamerV3 trains a **policy-dependent critic** (V^π(s)) - the expected return if you follow the **current actor** from state s.

**The problem**: Replay trajectories were collected under old policies (μ), so they're **off-policy** for the critic.

**The solution**: Combine two signals:
1. **Imagination rollouts**: Cheap on-policy training signal (but biased when world model is wrong)
2. **Replay data**: Real rewards/terminations for grounding (but off-policy actions)

The replay loss is a **stabilizer** (weighted 0.3), not the main training signal.

---

## What Replay Contributes

From replay you "steal":
- **Posterior latent states** (from real observations) - trustworthy
- **Real rewards** - trustworthy
- **Real continues/terminations** - trustworthy

---

## What "Value Annotation" Is

For each replay timestep (t), compute **one scalar**:
- **Imagination λ-return at the start of the dream rollout** starting from that real posterior state

This scalar is called a **value annotation** because it plays the role of a value term inside the replay λ-return recursion.

**Why?** Because replay futures are off-policy. The annotation makes replay targets "more on-policy" without trusting the replay action sequence.

---

## The Formula

Replay λ-returns with imagination annotations:
```
R_t^λ,replay = r_t^real + γ * c_t^real * ((1-λ) * V_t^annot + λ * R_{t+1}^λ,replay)
```

Where `V_t^annot = R_t^λ,imag` (imagination returns at timestep t).

---

## Implementation Changes

### 1. config.yaml - New Parameter
```yaml
# src/dreamer/conf/config.yaml
train:
  # Critic replay loss (DreamerV3 paper: βrepval=0.3)
  # This is a stabilizer, not the main training signal
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

**Key insight**: We need to collect imagination returns for all timesteps, then compute replay loss outside the timestep loop.

```python
# src/dreamer/trainer/core.py - train_models method

# Initialize accumulators for replay loss (NEW)
imagination_returns_list = []  # Will collect [T, batch] imagination returns
replay_critic_logits_list = []  # Will collect [T, batch, num_bins] critic logits

for t_step in range(T):
    # ... existing encoder and world model code ...

    # NEW: Compute critic logits on posterior states for replay loss
    # IMPORTANT: Detach posterior states to avoid pushing gradients into world model
    # CRITICAL: Do NOT use torch.no_grad() here - we want critic gradients
    replay_critic_logits = self.critic(h_z_joined.detach())
    replay_critic_logits_list.append(replay_critic_logits)  # Shape: [batch, num_bins]

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

    # Stack replay critic logits: [T, batch, num_bins]
    replay_critic_logits = torch.stack(replay_critic_logits_list, dim=0)

    # Compute replay lambda returns using imagination annotations
    # rewards: [batch, T] -> transpose to [T, batch]
    replay_rewards = self.rewards.transpose(0, 1)
    # continues: [batch, T] -> transpose to [T, batch]
    # IMPORTANT: Replay continues are already probabilities (1 - terminated)
    # Do NOT sigmoid them like logits
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

    # Apply mask to replay lambda returns
    replay_lambda_returns = replay_lambda_returns * replay_mask

    # Flatten for loss computation
    replay_critic_logits_flat = replay_critic_logits.view(-1, replay_critic_logits.size(-1))
    replay_lambda_returns_flat = replay_lambda_returns.detach().reshape(-1)

    # Twohot encode targets
    replay_targets = twohot_encode(replay_lambda_returns_flat, self.B)

    # Compute replay critic loss
    # IMPORTANT: Do NOT use torch.no_grad() here - we want critic gradients
    replay_loss = -torch.sum(
        replay_targets * F.log_softmax(replay_critic_logits_flat, dim=-1), dim=-1
    ).mean()

    # Add to total critic loss with proper weighting
    total_critic_loss += self.config.train.critic_replay_scale * replay_loss
```

### 4. core.py - Logging Enhancement

```python
# src/dreamer/trainer/core.py - log_metrics method

# Add replay loss to logging
if log_scalars:
    self.logger.add_scalar("loss/critic/replay", replay_loss.item(), step)
    self.logger.add_scalar("_key/loss_critic_replay", replay_loss.item(), step)
```

---

## Implementation Gotchas (Don't Break These!)

1. **Don't `torch.no_grad()` replay critic forward pass**
   - You want critic gradients to flow
   - Only detach the posterior states

2. **Detach posterior states when training critic from replay**
   - Prevents critic gradients from pushing into world model
   - `h_z_joined.detach()` is correct

3. **Be careful with continues**
   - Replay continues are already probabilities: `(1 - terminated).float()`
   - Do NOT sigmoid them like logits unless they are logits

4. **Replay loss is computed after collecting per-timestep annotations**
   - Must collect all imagination returns first
   - Then compute replay loss outside the timestep loop

5. **Replay loss is a stabilizer, not the main signal**
   - Weighted at 0.3 (much smaller than imagination loss at 1.0)
   - Provides grounding, not primary training

---

## Why This Works

1. **Imagination rollouts** provide lots of on-policy training signal (but can drift when world model is wrong)

2. **Replay data** provides real rewards/terminations for grounding (but actions are off-policy)

3. **Value annotations** make replay targets "more on-policy" without trusting the replay action sequence

4. **EMA regularizer** (already implemented) prevents "critic chases itself" oscillations during bootstrapping

5. **Combined**: Coverage (imagination) + Grounding (replay) = Stable learning

---

## Expected Behavior

- Critic trains on both imagined and real data
- Training targets are imagination-anchored to prevent off-policy drift
- More stable training by grounding the critic in real reward signals
- Faster convergence on tasks with hard-to-predict rewards
- Reduced oscillations due to EMA regularizer

---

## Compact Comment Block (for code)

```python
"""
Critic Replay Loss (DreamerV3):

The critic learns V^π(s) - expected return under current policy.
Replay trajectories are off-policy (collected under old actors).

Solution: Use imagination returns as value annotations for replay targets.

Replay λ-returns:
  R_t^λ,replay = r_t^real + γ * c_t^real * ((1-λ) * V_t^annot + λ * R_{t+1}^λ,replay)

Where V_t^annot = R_t^λ,imag (imagination return at timestep t).

This provides:
- Real rewards/terminations for grounding
- On-policy value annotations to avoid learning old policy value

Replay loss is a stabilizer (βrepval=0.3), not the main training signal.

Implementation notes:
- Detach posterior states (no encoder updates)
- Don't torch.no_grad() critic forward pass (want gradients)
- Replay continues are probabilities, not logits
- Compute after collecting all imagination returns
"""
```
