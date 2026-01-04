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

