# Critic Replay Loss Implementation Plan (DreamerV3)

## Overview
This plan describes how to implement the critic replay loss from DreamerV3, which trains the critic on both imagined trajectories and real replay data using imagination-anchored targets.

## Key Changes

### 1. Config.yaml Modification
Add the critic replay loss scale parameter (paper uses 0.3):

```yaml
# Training hyperparameters
train:
  # Critic replay loss (DreamerV3 paper: βrepval=0.3)
  critic_replay_scale: 0.3
```

### 2. Calculate Lambda Returns Enhancement
Modify `calculate_lambda_returns` to accept custom bootstrap values:

```python
def calculate_lambda_returns(
    rewards,
    values,
    continues,
    gamma,
    lam,
    num_steps,
    bootstrap_values=None,  # New optional parameter
):
    """
    Calculate lambda-returns (GAE-style) with optional custom bootstrap values.
    
    Args:
        ...
        bootstrap_values: Optional custom bootstrap values for final step
    """
    lambda_returns = []
    if bootstrap_values is not None:
        next_lambda_return = bootstrap_values
    else:
        next_lambda_return = values[-1]
    
    for i in reversed(range(num_steps)):
        reward_t = rewards[i]
        continue_prob_t = torch.sigmoid(continues[i])
        value_t = values[i]
        
        next_lambda_return = reward_t + gamma * continue_prob_t * (
            (1 - lam) * value_t + lam * next_lambda_return
        )
        lambda_returns.append(next_lambda_return)
    
    return torch.stack(lambda_returns).flip(0)
```

### 3. Critic-Only Loss Helper
Create a helper function in `losses.py` for critic-only loss:

```python
def compute_critic_loss(
    values_logits,
    lambda_returns,
    B,
    values_logits_ema=None,
    critic_ema_coef=1.0,
):
    """
    Compute critic loss using lambda returns as targets.
    
    Args:
        values_logits: Critic logits
        lambda_returns: Target lambda returns
        B: Bin tensor for twohot encoding
        values_logits_ema: EMA critic logits (for regularization)
        critic_ema_coef: EMA regularization coefficient
    """
    values_logits_flat = values_logits.view(-1, values_logits.size(-1))
    lambda_returns_flat = lambda_returns.detach().reshape(-1)
    critic_targets = twohot_encode(lambda_returns_flat, B)
    
    critic_loss = -torch.sum(
        critic_targets * F.log_softmax(values_logits_flat, dim=-1), dim=-1
    ).mean()
    
    if values_logits_ema is not None:
        ema_logits_flat = values_logits_ema.view(-1, values_logits_ema.size(-1)).detach()
        ema_probs = F.softmax(ema_logits_flat, dim=-1)
        ema_reg_loss = -torch.sum(
            ema_probs * F.log_softmax(values_logits_flat, dim=-1), dim=-1
        ).mean()
        critic_loss += critic_ema_coef * ema_reg_loss
    
    return critic_loss
```

### 4. Core Training Loop Modification
Update `core.py`'s `train_models` method to:

1. Compute posterior recurrent states from real data
2. Calculate replay lambda returns using imagination returns as annotations
3. Compute and add the replay loss to total critic loss

**Key steps:**
```python
# In core.py - after computing dreamed lambda returns
if self.config.train.critic_replay_scale > 0:
    # Compute posterior recurrent states from real data
    # (These are the actual states from the encoder)
    posterior_states = h_z_joined  # From encoder output
    
    # Use imagination lambda returns (R_lambda_imag) as bootstrap annotations
    replay_bootstrap_values = lambda_returns[0]  # Imagination returns at step t=0
    
    # Calculate lambda returns for real replay data
    replay_lambda_returns = calculate_lambda_returns(
        self.rewards,
        # We don't use critic's own values for replay bootstrap per paper
        torch.zeros_like(self.rewards),  # Dummy, since we use custom bootstrap
        (1 - self.terminated).float(),  # continues = 1 - terminated
        self.gamma,
        self.lam,
        self.sequence_length,
        bootstrap_values=replay_bootstrap_values,  # Imagination annotations
    )
    
    # Compute critic logits on real posterior states (detached to avoid encoder updates)
    replay_values_logits = self.critic(posterior_states.detach())
    
    # Get EMA critic values for replay loss regularization
    with torch.no_grad():
        replay_values_logits_ema = self.critic_ema(posterior_states.detach())
    
    # Compute replay loss
    replay_loss = compute_critic_loss(
        replay_values_logits,
        replay_lambda_returns,
        self.B,
        replay_values_logits_ema,
        self.critic_ema_regularizer,
    )
    
    # Apply mask and add to total critic loss
    mask_t = self.mask[:, t_step].mean()
    total_critic_loss += self.config.train.critic_replay_scale * replay_loss * mask_t
```

### 5. Logging Enhancement
Add replay loss to MLflow logging:

```python
# In log_metrics method
self.logger.add_scalar("loss/critic/replay", replay_loss.item(), step)
self.logger.add_scalar("_key/loss_critic_replay", replay_loss.item(), step)
```

## Verification Steps
1. Check that both critic loss components are present in training logs
2. Verify that replay loss is properly masked
3. Confirm that gradients don't flow from critic to encoder
4. Test with config.train.critic_replay_scale = 0.3
5. Compare training stability and performance

## Expected Benefits
- More stable critic training by grounding it in real reward signals
- Better generalization by using both imagined and real data
- Faster convergence in tasks with hard-to-predict rewards