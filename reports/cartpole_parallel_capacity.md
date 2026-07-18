# CartPole Parallel Capacity Profile

Date: 2026-07-18

Code baseline: `5dcfe74`

Profiler: `scripts/profile_cartpole_capacity.py`

## Decision

Run at most **three concurrent CartPole experiments**, with one collector and two
PyTorch intra-op CPU threads per experiment. Three-way concurrency gave the best
measured aggregate throughput while keeping progress reasonably fair across
seeds. Use two-way concurrency when the workstation is also under a substantial
interactive load.

Do not use four-way concurrency for the frozen benchmark. It remained within
RAM and VRAM limits, but two of four runs slowed to nearly half the rate of the
other two. That scheduling imbalance weakens both wall-clock efficiency and the
comparability of supposedly identical seed runs.

## Host

- CPU: Intel Core Ultra 7 265KF, 20 physical cores, no SMT exposed
- RAM: 31 GiB total; 22.8-23.2 GiB available before these probes
- GPU: Radeon RX 7900 XTX, 24 GiB VRAM
- Desktop VRAM baseline: 3.8-3.9 GiB used
- Software: Python 3.13.7, PyTorch 2.9.1+rocm6.4

The ROCm environment is isolated at `.venv-rocm`; CPU and ROCm extras were not
installed into the same environment.

## Workload

Each process ran the supported training entry point in dry-run mode with:

- state-only `CartPole-v1`
- `d_hidden=128`
- batch size `8`, sequence length `16`, replay burn-in `8`
- full actor/critic training from step zero
- one collector, replay ratio `1.0`, replay capacity `1000` episodes
- 120 training updates, deterministic seed equal to the run index
- no evaluation, checkpoints, retained MLflow run, or long training matrix

Dry-run mode removes artifact I/O but preserves the parent, collector, trainer,
replay, model-update, and shutdown paths. The profiler sampled each launcher and
all descendants once per second using `psutil`; GPU metrics came from
`rocm-smi`. Host available-memory reduction is the system-level memory signal.
Per-run USS is also recorded because summed RSS double-counts shared libraries.

## Results

All measured training commands exited successfully.

| Concurrent runs | Per-run updates/s | Aggregate updates/s | Group wall time | Peak host RAM delta | Peak VRAM delta | Mean CPU | Mean GPU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.98 | 4.98 | 44.8 s | 3.04 GiB | 0.59 GiB | 12.8% | 48.1% |
| 2 | 4.49, 4.58 | 9.07 | 47.8 s | 5.80 GiB | 1.26 GiB | 24.2% | 55.1% |
| 3 | 4.36, 4.35, 4.04 | **12.75** | 51.8 s | 9.40 GiB | 1.92 GiB | 34.7% | 56.6% |
| 4 | 3.44, 1.88, 1.88, 3.37 | 10.57 | 80.0 s | 11.48 GiB | 2.48 GiB | 46.0% | 72.4% |

Three-way concurrency achieved 2.56 times the single-run aggregate rate. The
slowest of the three runs was 19% below the single-run rate. Four-way
concurrency reduced aggregate throughput relative to three-way and introduced a
large per-run scheduling split, despite having enough memory.

### CPU thread cap

The repository does not currently set PyTorch thread counts. On this host, each
new process defaults to 20 intra-op threads. A single comparison run shows that
this default is actively harmful for this workload:

| Threads per process | Updates/s | Group wall time | Peak process-tree threads | Mean CPU | Mean GPU |
|---:|---:|---:|---:|---:|---:|
| 2 | 4.98 | 44.8 s | 21 | 12.8% | 48.1% |
| 20 | 1.68 | 91.5 s | 133 | 67.1% | 28.0% |

The 20-thread run was slower even though it ran after ROCm initialization. For
parallel runs, set all of the following to `2` before launching:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
```

These variables limit the CPU kernels used by PyTorch and numerical libraries.
They do not limit the number of operating-system processes: each experiment
still has a parent, collector, trainer, and multiprocessing resource tracker.

## Launch policy

1. Default to three concurrent runs for a three-seed CartPole comparison.
2. Use exactly one collector per run. Increasing collectors changes both the
   resource profile and the data-generation experiment.
3. Give every run a seed-specific Hydra output directory. Simultaneous runs must
   never share checkpoints, manifests, configuration snapshots, or MLflow state.
4. Before launch, require at least 18 GiB host RAM available and 12 GiB VRAM
   free. These thresholds preserve roughly 8 GiB host headroom over the measured
   three-run peak and ample GPU headroom.
5. During training, stop launching new work if available host RAM falls below
   8 GiB, swap usage grows by more than 2 GiB, or free VRAM falls below 8 GiB.
6. Do not overlap Pong and a three-way CartPole batch. Pong has a materially
   different pixel replay and VRAM/RAM profile and needs its own capacity check.
7. Treat four concurrent runs as unsupported until a longer probe explains and
   removes the observed scheduling unfairness.

## Limits of this profile

- Each capacity point is one short trial, not a statistical performance study.
- The workload excludes periodic evaluation, checkpoint writes, and MLflow I/O.
- It measures current-master model shapes. Historical commits may have different
  process or memory behavior and should receive a short canary before a full run.
- ROCm reports card-wide VRAM and utilization, so GPU deltas are aggregate rather
  than attributed to individual experiments.
- Long-run monitoring remains necessary. This profile establishes a conservative
  launch limit; it does not prove that memory use can never drift.

## Raw evidence

- `reports/profiling/cartpole_capacity_1x.json`
- `reports/profiling/cartpole_capacity_1x_threads20.json`
- `reports/profiling/cartpole_capacity_2x.json`
- `reports/profiling/cartpole_capacity_3x.json`
- `reports/profiling/cartpole_capacity_4x.json`
