# GPU Profiling Plan for SleepyDreamyV3

## Current Status

You have **coarse timing instrumentation** in `src/trainer.py` that prints:
```
[PROFILE] data: XX.X% | fwd: XX.X% | bwd: XX.X% | total: X.XXs/50 steps
```

This tells you which **phase** is the bottleneck.

---

## Step 1: Interpret Coarse Timing Results

Run training and observe the `[PROFILE]` output. Here's how to interpret:

| Result | Meaning | Next Action |
|--------|---------|-------------|
| **data > 50%** | Data loading/transfer is bottleneck | Profile `get_data_from_queue()`, check replay buffer |
| **fwd > 50%** | Forward pass is bottleneck | Profile encoder vs RSSM vs dream sequence |
| **bwd > 50%** | Backward pass is bottleneck | Check gradient accumulation, model size |
| **Balanced** | No single bottleneck | Increase batch size, then re-profile |

---

## Step 2: Fine-Grained Profiling (Once You Know the Phase)

### Option A: PyTorch Profiler (works with ROCm)

Add this temporarily to profile one training step:

```python
# In train_models(), wrap ONE iteration:
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # ... one training iteration ...

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### Option B: rocprof (AMD-specific, lower level)

```bash
rocprof --stats python -m src.main --config env_configs/cartpole.yaml
```

### What to Look For in Profile Output

```
Name                      CPU Time   CUDA Time   Calls
-------------------------  --------   ---------   -----
aten::mm                   10ms       50ms        200    <- Many small matmuls
aten::convolution          5ms        30ms        100    <- Encoder calls
aten::copy_                20ms       20ms        50     <- Memory transfers!
```

**Red flags:**
- High `Calls` count with low time each = kernel launch overhead
- `aten::copy_` or `aten::to` high = excessive CPU<->GPU transfers
- Big gap between CPU and CUDA time = sync issues

---

## Step 3: Common Optimizations for RSSM Architecture

### If data loading is slow:
1. Check `replay_buffer.sample()` - is it doing unnecessary copies?
2. Move resize to GPU: currently `resize_pixels_to_target` may be on CPU
3. Use pinned memory: `torch.from_numpy(...).pin_memory().to(device, non_blocking=True)`

### If forward pass is slow:
1. **Batch the encoder** - process all timesteps at once:
   ```python
   # Instead of loop:
   # for t in range(T): encoder(pixels[:, t])

   # Do this:
   all_posteriors = encoder(pixels.view(B*T, C, H, W))
   all_posteriors = all_posteriors.view(B, T, -1)
   # Then loop through RSSM only
   ```

2. **Increase batch size** - you're at 16% VRAM, aim for 60-70%:
   ```yaml
   # env_configs/cartpole.yaml
   train:
     batch_size: 32  # or 64, 128 - experiment
   ```

3. **Check for sync points** - `.item()` calls force GPU sync:
   ```python
   # BAD (inside loop):
   wm_loss_components[key] += wm_loss_dict[key].item()

   # BETTER (accumulate tensors, convert at end):
   wm_loss_components[key] += wm_loss_dict[key]
   # ... after loop ...
   wm_loss_components = {k: v.item() for k, v in wm_loss_components.items()}
   ```

### If backward pass is slow:
1. Check if `retain_graph=True` is being used unnecessarily
2. Gradient checkpointing for memory-bound cases
3. Mixed precision training: `torch.cuda.amp.autocast()`

---

## Step 4: Memory Profiling

Check VRAM usage:
```bash
watch -n1 rocm-smi
```

If VRAM is low (like your 16%), you're leaving performance on the table.
Increase batch size until you hit 60-80% VRAM.

Profile memory allocation:
```python
print(torch.cuda.memory_summary())
```

---

## Step 5: Advanced - Profile Individual Components

If forward pass dominates, add sub-timers:

```python
# Inside the t_step loop:
t_enc = time.perf_counter()
posterior_logits = self.encoder(obs_t)
torch.cuda.synchronize()
t_encoder_acc += time.perf_counter() - t_enc

t_wm = time.perf_counter()
(...) = self.world_model(posterior_dist, action_t)
torch.cuda.synchronize()
t_worldmodel_acc += time.perf_counter() - t_wm

t_dream = time.perf_counter()
(...) = self.dream_sequence(...)
torch.cuda.synchronize()
t_dream_acc += time.perf_counter() - t_dream
```

---

## Quick Reference: ROCm Tools

| Tool | Use Case |
|------|----------|
| `rocm-smi` | GPU utilization, VRAM, temp |
| `rocprof --stats` | Kernel-level timing |
| `torch.profiler` | PyTorch operation timing |
| `torch.cuda.memory_summary()` | Memory allocation breakdown |

---

## The Mental Model

```
GPU Efficiency = Time Doing Math / Total Time

Low efficiency causes:
- Kernel launch overhead (many small ops)
- Memory transfers (CPU <-> GPU)
- Sync points (.item(), print, logging)
- Small batch sizes (GPU underutilized)

Your goal: Big batches, few kernel launches, minimal transfers
```

---

## Next Steps

1. Run training, observe `[PROFILE]` output
2. Identify which phase dominates
3. Apply targeted profiling to that phase
4. Make ONE change, re-measure
5. Repeat
