# Research Protocol

One hypothesis per run: "Change X → expect Y" with one primary metric.
Config layers: `src/config.py` (base) + `env_configs/*.yaml` (env overrides) + CLI (one-off).
Gate awareness: architecture/env changes = new model family; training dynamics = experimentation zone.
Git commits for logic changes; run artifacts (`runs/`) carry outcomes.

# Cartpole
### 12-30-25
Initial WM training. Expect WM to converge for cartpole. Will investigate AC later

### 12-31-25 Run 1 (12-31_0623)
- Disabled torch.compile (5.4x speedup on ROCm)
- Entropy: 0.1 → 0.001 cosine decay
- Step 2500: avg 9.2 steps (similar to old run at 7500)
- Conclusion: torch.compile was bottleneck, not entropy

### 12-31-25 Run 2 (01-01_0935)
- Hypothesis: Higher entropy start (0.15) → more exploration → better policy
- Change: entropy_coef_start 0.1 → 0.15
- Results:
  | Step  | Avg Length |
  |-------|------------|
  | 2500  | 9.0        |
  | 7500  | 9.8        |
  | 15000 | 9.4        |
  | 25000 | 16.8       |
- Random policy baseline: ~20 steps
- **Conclusion: Policy is WORSE than random. Actor not learning.**
- Possible causes: actor gradients not flowing, dream states unrepresentative, lambda returns bug

### 01-02-26 Run 3 (01-02_0957)
- Hypothesis: Lambda returns bug (using V(s_t) instead of V(s_{t+1})) slows learning → fix should accelerate convergence
- Change: Fixed `calculate_lambda_returns` to use `dreamed_values[i+1]` instead of `dreamed_values[i]`
- Baseline from Run 2: 10k→10.0, 25k→16.4, 45k→21.0 avg episode length
- Resumed from 25k checkpoint with fix applied
- Results:
  | Step  | Avg Length | Notes |
  |-------|------------|-------|
  | 30000 | 27.1       | Already beats old 45k (21.0) |
  | 35000 | 28.2       | Improving but slowly |
- **Conclusion: Fix works! Learning faster than before, but still slow.**
- Analysis: d_hidden=64 may be too small, entropy cosine decay may be suboptimal

### 01-03-26 Profiling Results
- Hypothesis: Larger model capacity → slower training
- Tested: d_hidden 64, 128, 256 (120s each, clean runs)
- Results:
  | d_hidden | Steps/sec |
  |----------|-----------|
  | 64       | 0.54      |
  | 128      | 0.59      |
  | 256      | 0.58      |
- **Conclusion: Model size NOT the bottleneck.** CPU/data loading limits speed.
- Can use larger models without speed penalty.

### 01-03-26 Run 5 (01-03_1253)
- Hypothesis: Larger model + constant entropy → faster convergence
- Changes:
  1. d_hidden: 64 → 256 (4x capacity)
  2. Entropy: cosine decay (0.15→0.001) → constant 0.005 (DreamerV3 style)
- Training from scratch, 50k steps
- Baseline comparison: Run 3 with d_hidden=64 reached 28.2 avg at 35k

### 01-03-26 Collector Optimization
- Hypothesis: Resizing images early (in collector) → faster training
- Problem: Rendering 400x600 images, then resizing to 64x64 in trainer → massive waste
- Change: Added cv2 resize in collector immediately after env.step()
- Results:
  | Config      | Before | After  | Speedup |
  |-------------|--------|--------|---------|
  | d_hidden=256| 0.58   | 4.77   | **8.2x**|
- **Conclusion: Early resize is huge win.** CPU/data pipeline was the bottleneck.
- GPU now better utilized (was 37%, should be higher now)

### 01-03-26 Batch Size Test
- Hypothesis: Larger batch size → better GPU utilization → faster training
- Tested: batch_size 8 vs 16
- Results:
  | Batch Size | Steps/s | Throughput |
  |------------|---------|------------|
  | 8          | 4.77    | 38.2       |
  | 16         | 1.77    | 28.3       |
- **Conclusion: batch_size=8 is optimal.** Larger batches cause data starvation.
- Collector can't fill replay buffer fast enough for batch_size=16

### 01-03-26 Replay Buffer Implementation
- Problem: batch_size=16 caused data starvation, batch_size=16 + 2 collectors caused hangs
- Root cause: No replay buffer. Trainer blocked on queue.get() for each episode.
- Solution: Implemented EpisodeReplayBuffer with background drain thread
- Results with replay buffer:
  | Config | Steps/s | Notes |
  |--------|---------|-------|
  | batch_size=8, 1 collector | 5.08 | Slight improvement |
  | batch_size=16, 1 collector | 3.5-3.7 | 2x faster than before (was 1.77) |
  | batch_size=16, 2 collectors | 2.9 | Slower - multi-collector overhead |
- **Conclusion: Replay buffer enables larger batches.** But 2 collectors adds overhead without benefit.

### 01-03-26 Fixed-Length Subsequence Sampling
- Problem: Variable episode lengths caused batch truncation to shortest episode (~10 steps)
- With batch_size=32, high chance of very short episode → all batches truncated to ~8-10 steps
- Solution: Sample fixed-length subsequences (like DreamerV3) instead of whole episodes
- DreamerV3 uses sequence_length=64 for better temporal credit assignment
- Results:
  | Config | Steps/s | Transitions/s | Notes |
  |--------|---------|---------------|-------|
  | Old (truncate to min ~10) | 5.0 | 400 | Fast but wastes data |
  | New (fixed seq_len=25) | 1.4 | 280 | Slower but better gradients |
- **Tradeoff**: Longer sequences = slower steps/s but better learning (more temporal context)

### 01-03-26 Clean Profiling (after killing zombie processes)
- Previous tests had 11+ zombie trainer processes consuming 30GB+ RAM - results were skewed
- Clean test with seq_len=25, batch_size=8:
  - 100 steps in 60s = **1.67 steps/s**
  - Throughput: 8 × 25 × 1.67 = **333 transitions/s**
- Tested seq_len=64: Only 1 step in 60s (CartPole episodes ~25 steps, 63% wasted on padding)
- **Conclusion**: seq_len should match environment episode length. For CartPole, seq_len=25 is optimal.
- Final throughput comparison:
  | Approach | Steps/s | Transitions/s | Notes |
  |----------|---------|---------------|-------|
  | Old (truncate to ~10) | 5.0 | 400 | Wastes long episode data |
  | New (fixed seq_len=25) | 1.67 | 333 | Better gradients, 17% slower |
- **Decision**: Keep fixed-length sampling. 17% throughput loss is acceptable for better learning.

### 01-03-26 Vectorized Data Loading Bug Fix
- Problem: batch_size=32 was SLOWER than batch_size=16 (1.38 vs 2.34 steps/s)
- Root cause: `get_data_from_queue()` had O(batch_size) Python loop:
  ```python
  for pixels, ... in subsequences:  # Called batch_size times!
      pixels_resized = resize_pixels_to_target(...)  # Expensive per-item call
  ```
- Fix: Vectorized the data loading - stack numpy arrays first, single tensor conversion, single batched resize
- Results after fix (seq_len=25):
  | batch_size | steps/s | trans/s |
  |------------|---------|---------|
  | 32         | 3.08    | 2,464   |
  | 64         | 2.24    | 3,584   |
  | 128        | 1.31    | 4,192   |
- **Conclusion**: Larger batch sizes now scale properly. Higher batch = higher throughput.

### 01-03-26 Batch Size vs Sample Efficiency Analysis
- Question: Why does DreamerV3 use B=16, T=64 instead of maximizing batch size?
- DreamerV3 hyperparameters: B=16, T=64 → 1,024 trans/step
- Our config: B=128, T=25 → 3,200 trans/step (3x more throughput)
- **Key insight: Throughput != Learning speed**
  - Larger batches have diminishing returns for gradient quality
  - Gradient noise (from small batches) acts as regularization
  - In RL, larger batches average over stale replay data
- DreamerV3 optimized for **sample efficiency** (learn most from limited data)
- We're optimizing for **wall-clock speed** (samples are cheap in simulation)
- **Conclusion**: trans/step is misleading. True metric is **time-to-solve**.
- **TODO**: Run actual training to compare time-to-solve across batch sizes.

### 01-05-26 Profiling + Logging + Data Path Tweaks
- Added Torch profiler (chunked traces) and compile flag; profiled trainer/collector.
- Reduced logging overhead: scalars every 25 steps, images/video every 250 steps.
- Swapped categorical KL to manual logit-based form; set Categorical validate_args=False (old code commented).
- Data pipeline: skip resize if already at target size; move uint8 to GPU then cast to float to cut transfer overhead.

### 01-05-26 Run (cartpole_b8, warmup=5k, step~7.5k)
- Observed: actor entropy collapsed to ~0; dream reward mean oscillating; value mean/std exploding; KL raw ~0.46 with spikes.
- Interpretation: likely AC divergence shortly after warmup; WM may be weak and dreams collapse to vertical pole.
- Recommendation: stop unless avg episode length is still improving; next run lower actor/critic LR, shorten num_dream_steps, add grad clipping; log raw KL and eval returns.

### 01-09-26 State-Only Training Mode
- **Hypothesis**: Pixel observations add unnecessary complexity for simple envs like CartPole where state vector is sufficient.
- **Observation**: WM was learning cart position from pixels but not pole angle (pole is ~2-3 pixels wide at 64x64). Reward loss stuck at ~0.5 (expected - twohot entropy floor).
- **Implementation**: Added `use_pixels: false` config option
  - New `StateOnlyEncoder` class (MLP only, no CNN)
  - New `StateOnlyDecoder` class (MLP only, no CNN)
  - Conditional paths in trainer, environment, replay buffer
  - Collector throttling (backpressure when queue >80% full) to prevent wasteful over-collection
- **Config**: `env_configs/cartpole_state_only_small.yaml`
  - `use_pixels: false`
  - `d_hidden: 64` (smaller - no images to encode)
  - `b_start: -5, b_end: 6` (smaller bin range for CartPole's value scale ~0-100)
- **Results**: Training runs without explosion. Much faster data collection (no rendering).
- **Files modified**: config.py, encoder.py, decoder.py, world_model.py, trainer_utils.py, environment.py, trainer.py, replay_buffer.py, main.py, evaluate.py
- **Next**: Verify CartPole solves faster with state-only mode vs pixel mode.

### 01-15-26 Free Bits Gradient Bug
- **Problem**: State-only CartPole trained for 60k steps, avg episode length stuck at ~9.5 (worse than random ~25)
- **Investigation**:
  1. Dream rollouts showed continue predictor crashing to 0.001 after 3 steps
  2. Prior z entropy ~1.2 (diffuse), posterior z entropy ~0.1 (sharp) - they only agree ~40-60%
  3. Dynamics loss stuck at exactly 1.0 (free bits threshold) for entire 60k steps
  4. Raw KL was ~0.84, always below threshold
- **Root cause**: `torch.max(free_bits, l_dyn_raw)` returns gradient=0 when raw < threshold
  ```python
  # Bug: gradient is 0 when l_dyn_raw < 1.0
  l_dyn = torch.max(torch.tensor(free_bits), l_dyn_raw)
  ```
  - Dynamics predictor received **zero gradient** for 60k steps
  - Prior never learned to match posterior
  - Continue predictor saw OOD inputs during dreaming → predicted termination
- **Fix**: Straight-through estimator for free bits
  ```python
  # Forward: max(free_bits, raw) for loss value
  # Backward: gradient flows through raw
  l_dyn = l_dyn_raw + (free_bits - l_dyn_raw).clamp(min=0).detach()
  ```
- **Files modified**: trainer.py (lines 926-930)

### 01-15-26 Post-Fix Training Results
- **Result**: Fix works! Training now learns.
  - Before fix: stuck at 9.5 avg episode length (worse than random ~25)
  - After fix: 60-70 avg episode length at 7500 steps
  - KL dynamics raw: 0.84 → 0.001 (prior now matches posterior)
- **Remaining issue**: Oscillating around 60, not converging to 500 (solved)
  - Critic underestimating: predicts value ~14, should be ~38 for 48-step episodes
  - Policy finds good actions, then drifts away
- **Diagnosis**: Hyperparameter tuning needed, not a bug
  - Actor entropy getting low (~0.26) - less exploration
  - Dream horizon (15 steps) may be too short for CartPole (can be 500 steps)
  - Possible fixes: increase entropy coef, lower actor LR, longer dream horizon
- **Conclusion**: Implementation works. Solving CartPole requires tuning, which is a separate skill from debugging.
- **Status**: Project archived. Core learning: `torch.max` kills gradients below threshold; use straight-through estimator.

### 01-16-26 Actor-Critic Hyperparameter Sweep
- **Setup**: 10k bootstrap WM checkpoint, dreamer mode (frozen WM), 2.5k steps per config
- **Sweep**: actor_lr × actor_entropy_coef (3×4 = 12 configs)
- **Results**:

| LR | Entropy | Peak Ep Len | Final Ep Len | Pattern |
|-----|---------|-------------|--------------|---------|
| 3e-05 | 0.001 | **147.4** | 123.1 | Peak then decline |
| 3e-05 | 0.005 | **134.4** | 110.4 | Peak then decline |
| 3e-05 | 0.01 | 131.1 | 131.1 | Incomplete |
| 3e-05 | 0.03 | TBD | TBD | TBD |
| 1e-04 | 0.001 | TBD | TBD | TBD |
| 1e-04 | 0.005 | TBD | TBD | TBD |
| 1e-04 | 0.01 | TBD | TBD | TBD |
| 1e-04 | 0.03 | TBD | TBD | TBD |
| 1e-05 | 0.001 | 96.8 | 96.8 | Stable plateau |
| 1e-05 | 0.005 | 99.6 | 99.6 | Stable plateau |
| 1e-05 | 0.01 | 96.8 | 96.8 | Stable plateau |
| 1e-05 | 0.03 | 95.0 | 92.7 | Stable plateau |

- **Key Observations**:
  1. **LR dominates, entropy has minimal effect**: 0.001 vs 0.03 entropy showed no meaningful difference within same LR
  2. **Peak-then-decline pattern at high LR**: LR=3e-05 learns fast but destabilizes after peaking
  3. **Stable but slow at low LR**: LR=1e-05 never declines but plateaus around ~97
  4. **The decline is the problem to solve**: Best config (3e-05, 0.001) peaked at 147 but fell to 123

- **Root Cause Analysis**:
  - As returns grow (20→120), gradient magnitude grows proportionally
  - Without return normalization, larger returns = larger actor updates = overshoot
  - High LR amplifies this effect → policy finds good region, then overshoots out of it
  - Low LR masks the problem (smaller steps, less overshoot) but learns too slowly

- **DreamerV3 features present**: symlog, symexp, twohot, advantage normalization
- **DreamerV3 features missing**: return normalization, critic EMA, gradient clipping (AGC)

### 01-17-26 Return Normalization Implementation
- **Hypothesis**: Return normalization will stabilize high-LR training, preventing peak-then-decline
- **Implementation**: EMA-based return scale tracking (per DreamerV3 paper)
  - Track 5th/95th percentile of lambda returns via EMA (decay=0.99)
  - Normalize advantage by `max(1, S95 - S5)` - only scale down, never up
  - `max(1, ...)` prevents noise amplification under sparse rewards
- **Files modified**: trainer/core.py, trainer/losses.py
- **Test config**: Same as best sweep run (lr=3e-5, entropy=0.001, 2.5k steps)
- **Expected result**: Similar fast rise, stable plateau instead of decline

### 01-17-26 Run (01-17_141742) - Return Normalization Test
- **Config**: lr=3e-5, entropy=0.001, 10k steps (5k warmup + 5k AC)
- **Results**:

| Step | Avg Ep Len | Dream Value | KL Rep | Actor Loss | Critic Loss | WM Reward |
|------|------------|-------------|--------|------------|-------------|-----------|
| 5500 | ~80 (peak) | ~5 | 0.0025 | 0.20 | 0.46 | 0.630 |
| 6500 | ~55 | ~10 | 0.0015 | 0.18 | 0.44 | 0.635 |
| 7500 | ~47 (trough) | ~14 | 0.0015 | 0.17 | 0.43 | 0.645 |
| 8250 | 49.6 | 16.45 | 0.0001 | 0.166 | 0.426 | 0.651 |
| 10000 | ~50 | ~17 | ~0 | 0.14 | 0.41 | 0.67 |

- **Observations**:
  1. **Peak-then-collapse**: Episode length peaked at ~80 (step 5500), then collapsed to ~50
  2. **Delusional dreaming**: Dream value tripled (5→17) while episode length dropped 40%
  3. **KL rep → 0**: Collapsed from 0.0025 to 0.0001 (suspicious - too low)
  4. **WM reward loss rising**: 0.63 → 0.67 (WM struggling with new states)

- **Root Cause Analysis**: WM-Actor lag
  - During warmup (0-5k): WM learned "random policy dynamics" (short episodes)
  - AC kicks in (5k-6.5k): Policy improves fast, episodes 25→80
  - Collapse (6.5k+): AC outran WM, dreams became unreliable
  - Actor/Critic loss plateaued (can't learn from stale WM predictions)

### 01-17-26 DreamerV3 Implementation Audit
- **Discovery**: Our implementation deviated from DreamerV3 paper defaults

| Parameter | Ours | DreamerV3 | Impact |
|-----------|------|-----------|--------|
| βrep | **0.99** | **0.1** | 10x too high → KL collapse |
| βdyn | 0.99 | 1.0 | Minor |
| βpred | 0.99 | 1.0 | Minor |
| unimix | **Missing** | **1%** | Categorical collapse risk |
| γ | 0.99 | 0.997 | Ours more myopic |
| η (entropy) | 1e-3 | 3e-4 | Ours 3x higher |
| H (horizon) | 15 | 15 | Match |
| free nats | 1 | 1 | Match |

- **Critical Issues**:
  1. **βrep = 0.99 should be 0.1**: Representation loss 10x too high → pushes posterior toward prior too aggressively → latent collapse → KL rep → 0
  2. **Unimix missing**: 1% uniform mixture prevents categorical collapse by maintaining gradient flow

- **Fixes Implemented**:
  ```python
  # conf/config.yaml
  beta_dyn: 1.0
  beta_rep: 0.1  # was 0.99
  beta_pred: 1.0

  # src/trainer/math_utils.py - new function
  def unimix_logits(logits, unimix_ratio=0.01):
      probs = F.softmax(logits, dim=-1)
      uniform = torch.ones_like(probs) / num_classes
      probs_mixed = (1 - unimix_ratio) * probs + unimix_ratio * uniform
      return torch.log(probs_mixed + 1e-8)

  # Applied to: posterior, prior (dreams), prior (KL computation)
  ```

- **Additional Fix**: On-demand collection
  - Problem: Collector generated 1M+ episodes while trainer only used ~10k
  - Fix: Collector waits when queue >80% full instead of brief sleep
  - Benefit: Reduces CPU waste without starving GPU

### 01-17-26 WM-Actor Lag Analysis (Theoretical)
- **Core Problem**: Actor-critic learns faster from dreams than WM adapts to new state distribution
- **Timeline**:
  1. Warmup: WM learns random policy dynamics
  2. AC starts: Exploits WM, improves fast, reaches new states
  3. Lag: WM still has "random world" model, predictions become stale
  4. Collapse: AC learns wrong things from bad dreams

- **Detection Signals** (for future adaptive training):
  - Episode length drop (works for dense rewards)
  - WM loss spike on recent data (works for sparse rewards too)
  - KL divergence spike (model surprise)

- **Potential Interventions** (not yet implemented):
  | Approach | Description |
  |----------|-------------|
  | WM update ratio | Train WM 2-4x per AC update |
  | AC throttling | Update AC every N steps |
  | Surprise trigger | Boost WM when recent loss spikes |
  | Asymmetric LR | WM 1e-4, Actor 3e-5 |

- **Next Steps**: Test with βrep=0.1 and unimix fixes before adding adaptive mechanisms
