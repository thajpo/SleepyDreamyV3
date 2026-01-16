# DreamerV3 World Model Training - Learnings Archive

**Date Range:** December 29, 2024 - January 3, 2025  
**Environment:** CartPole-v1 / LunarLander-v3  
**Total Runs Analyzed:** 123 runs before cleanup

---

## Executive Summary

Over ~6 days of experimentation, we progressed from broken reward scaling and small model capacity to a stable training pipeline. Key discoveries:

1. **Model capacity matters**: `d_hidden=256` dramatically outperforms 64/128 (10-40x better pixel reconstruction)
2. **World model converges fast**: 90% of improvement happens in first 500-1000 steps
3. **Batch size tradeoffs**: Smaller batches (8) converge to lower loss; larger batches (64-128) more efficient per-step
4. **Bootstrap is critical**: 10,000 bootstrap steps required before actor training
5. **Reward prediction is hard**: Plateaus at ~0.6 loss - the dominant challenge

---

## Day-by-Day Evolution

### December 29 - Initial Debugging

**Runs:** 8 | **Best Run:** 12-29_0710 (10K steps, 545MB checkpoint)

**Key Events:**
- Early runs (0649-0702): Dream rewards massively broken (-1M to +470K instead of [-100, +200])
- Fixed reward scaling between 0702 and 0704
- Discovered `d_hidden=256` >> `d_hidden=64`
- First successful long run (10K steps)

**Critical Bug Fixed:** Reward scaling/normalization issue causing 10^6 scale errors

**Learnings:**
- KL divergence should grow from ~0.001 to ~0.8 during training (healthy latent space)
- Actor entropy collapsed to 0.001 - suggests need for higher `actor_entropy_coef`

### December 30 - Transition to Actor/Critic Training

**Runs:** 18 | **Best Run:** 12-30_1400 (11,837 steps, functioning actor)

**Key Events:**
- All configs identical except `max_train_steps`
- Run 12-30_0929 had exploding values (bug: dream value reached 8.9e7)
- Fixed by 12-30_1400 - stable actor/critic training

**Performance Baseline Established:**
| Metric | Converged Value |
|--------|-----------------|
| WM Loss | ~2.6 |
| Pixel Loss | ~0.008 |
| Reward Loss | ~0.62-0.68 |
| Critic Loss | ~0.57 |

**Learnings:**
- World model converges in ~1000 steps on CartPole
- Training speed: 0.72 steps/sec with actor/critic (vs 2.61 WM-only)
- Dream rewards stabilized to 0.78 → 1.00 trajectory

### December 31 - Entropy Coefficient Experiments

**Runs:** 2 | **Best Run:** 12-31_0801 (9,396 steps)

**Key Finding:** Higher entropy coefficient (0.15 vs 0.10) performed better:
- Better value estimation (2.25 vs 2.10)
- Lower KL divergence (0.58 vs 0.69)
- More confident policy despite higher regularization

**Architecture Settings Stabilized:**
```json
{
  "d_hidden": 64,
  "rnn_blocks": 4,
  "latent_categories": 16,
  "sequence_length": 25,
  "dream_steps": 15,
  "batch_size": 8,
  "all_lr": 1e-4
}
```

### January 1-2 - Extended Training Validation

**Runs:** 4 | **Best Run:** 01-02_0957 (62,790 steps)

**Key Events:**
- Longest successful training runs
- Validated bootstrap_steps=10,000 is essential (vs 100 which failed)
- Training speed improved: 0.46 steps/sec (up from 0.34)

**01-02_1853 Experiment:** Tested reduced warmup
- `actor_entropy_coef`: 0.01 → 0.001 (10x lower)
- `bootstrap_steps`: 10000 → 100
- **Result:** Failed - pixel loss 0.64 (vs 0.008 baseline)

**Conclusion:** Bootstrap phase cannot be shortened significantly.

### January 3 Morning - Model Size Sweep

**Runs:** 10 | **Best Run:** 01-03_1343 (50K steps, 11GB checkpoints)

**d_hidden Comparison:**
| d_hidden | Pixel Loss (converged) | Checkpoint Size |
|----------|------------------------|-----------------|
| 64 | 0.40 | 13MB |
| 128 | 0.45 | ~50MB |
| 256 | **0.01** | 514MB |

**Key Finding:** 256-dim models achieve **10-40x better pixel reconstruction**

**Training Saturation:**
- Loss plateau reached by ~2000-3000 steps
- 50K steps provides marginal improvement over 3K
- Consider early stopping for hyperparameter search

### January 3 Afternoon - Parallel Collection & Replay Buffer

**Runs:** 9 | **Best Run:** 01-03_1507 (28K steps, loss 2.606)

**New Parameters Tested:**
- `num_collectors=2` - parallel environment sampling
- `replay_buffer_size=1000`
- `min_buffer_episodes=64`

**Findings:**
- Batch size 8 enabled longest stable training
- `num_collectors=2` inconclusive (runs too short)
- Replay buffer params may have caused instability (run 1757 showed high loss)

**Loss Component Analysis (converged):**
| Component | Value |
|-----------|-------|
| Pixel | 0.014 |
| Vector | 0.0002 |
| Reward | 0.617 |
| Continue | 0.008 |
| KL (dynamics/rep) | 0.42 |

### January 3 Evening - Batch Size Sweep

**Runs:** 32 | **Best Run:** 01-03_1901 (2527 steps, loss 2.28)

**Batch Size Performance:**
| Batch Size | Avg Final Loss | Steps Efficiency |
|------------|----------------|------------------|
| 8 | 2.7-2.9 | Best absolute |
| 16 | 2.7-3.0 | Balanced |
| 32 | 2.8-3.1 | Good for short runs |
| 64-128 | 2.8-2.9 | Promising per-step efficiency |

**Key Insights:**
- Smaller batches (8) converge to lower loss but need more steps
- Larger batches (64-128) competitive loss/step - worth exploring
- Minimum viable steps: ~100+ for meaningful convergence
- `seq_len=64` (run 1857) performed worse - needs different hyperparameters

---

## Final Recommended Configuration

Based on all experiments, the optimal configuration for CartPole-v1:

```json
{
  "general": {
    "device": "cuda",
    "train_world_model": true
  },
  "models": {
    "d_hidden": 256,
    "encoder": {
      "cnn": {
        "num_layers": 4,
        "stride": 2,
        "kernel_size": 2,
        "activation": "sigmoid",
        "target_size": [64, 64]
      },
      "mlp": {
        "hidden_dim_ratio": 8,
        "n_layers": 3,
        "latent_categories": 16
      }
    },
    "rnn": {
      "n_blocks": 4
    }
  },
  "train": {
    "sequence_length": 25,
    "max_train_steps": 5000,
    "num_dream_steps": 15,
    "gamma": 0.99,
    "lam": 0.95,
    "wm_lr": 0.0001,
    "critic_lr": 0.0001,
    "actor_lr": 0.0001,
    "weight_decay": 1e-6,
    "batch_size": 8,
    "actor_warmup_steps": 1000,
    "bootstrap_steps": 10000,
    "actor_entropy_coef": 0.01,
    "beta_dyn": 0.99,
    "beta_rep": 0.99,
    "beta_pred": 0.99
  }
}
```

---

## Open Questions & Future Experiments

1. **Reward Prediction Plateau**: Loss stuck at ~0.6 - investigate architecture changes
2. **Actor Entropy Collapse**: Consider entropy scheduling (decay from high to low)
3. **Batch Size Scaling**: Test batch=128 with learning rate scaling (LR * sqrt(batch/8))
4. **Sequence Length**: seq_len=64 needs corresponding batch size reduction
5. **Checkpoint Efficiency**: 514MB per save - consider model compression
6. **Multi-collector**: Needs longer runs to evaluate `num_collectors=2`

---

## Best Checkpoints to Preserve (from deleted runs)

If migrating to new system, consider preserving:

| Run | Checkpoint | Why |
|-----|------------|-----|
| 01-02_0957 | step 62500 | Longest trained, best metrics |
| 01-03_1343 | step 50000 | Best d_hidden=256 run |
| 01-03_1507 | step 27500 | Best afternoon session |

---

*Document generated: January 4, 2025*  
*Runs deleted: All non-01-04 runs (57 directories)*
