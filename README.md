# SleepyDreamyV3

A PyTorch implementation of [DreamerV3](https://arxiv.org/abs/2301.04104) - a model-based reinforcement learning agent that learns world models and uses imagination for policy optimization.

## Features

- **World Model Learning**: Learns environment dynamics from experience using RSSM (Recurrent State-Space Model)
- **Imagination-Based Training**: Trains actor-critic policies entirely within the learned world model
- **State-Only & Pixel Modes**: Supports both low-dimensional state vectors and pixel observations
- **Hydra Configuration**: Flexible configuration management with YAML files and CLI overrides
- **Multiprocessing Architecture**: Separate collector and trainer processes for efficient GPU utilization
- **MLflow Logging**: Real-time training metrics, visualizations, and experiment tracking

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   Collector     │────▶│  Replay Buffer   │
│  (Environment)  │     └────────┬─────────┘
└─────────────────┘              │
                                 ▼
                    ┌────────────────────────┐
                    │       Trainer          │
                    │  ┌──────────────────┐  │
                    │  │   World Model    │  │
                    │  │ (Encoder, RSSM,  │  │
                    │  │  Decoder, Reward,│  │
                    │  │  Continue)       │  │
                    │  └──────────────────┘  │
                    │           │            │
                    │           ▼            │
                    │  ┌──────────────────┐  │
                    │  │  Dream Rollouts  │  │
                    │  │  (Imagination)   │  │
                    │  └──────────────────┘  │
                    │           │            │
                    │           ▼            │
                    │  ┌──────────────────┐  │
                    │  │  Actor-Critic    │  │
                    │  └──────────────────┘  │
                    └────────────────────────┘
```

## Installation

Requires Python 3.13+.

```bash
# Clone the repository
git clone https://github.com/thajpo/SleepyDreamyV3.git
cd SleepyDreamyV3

# Lightweight CPU development environment
uv sync --extra cpu
```

PyTorch is an explicit environment choice rather than an implicit platform
guess. Use the CPU extra for development and hosted CI. For AMD training,
create a separate environment so ROCm packages never inflate the CPU install:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rocm uv sync --extra rocm
```

ROCm 6.4 is the configured workstation backend. Set
`HSA_OVERRIDE_GFX_VERSION` only when the local GPU requires it. CUDA is not a
locked project environment yet; add it as a separate accelerator extra rather
than replacing the CPU or ROCm contract.

## Quick Start

### Training

```bash
# Full training with default config (CartPole, state-only)
uv run --extra cpu dreamer-train

# Override parameters via CLI
uv run --extra cpu dreamer-train \
  train.actor_entropy_coef=1e-3 train.max_train_steps=50000

# Smoke test / dry run (no MLflow, no checkpoints, temp directory)
uv run --extra cpu dreamer-train \
  general.dry_run=true general.device=cpu \
  train.max_train_steps=1 train.min_buffer_episodes=2 \
  train.batch_size=2 train.sequence_length=4 \
  train.replay_burn_in=1 train.eval_every=0
```

Resume into a new output directory while restoring model and optimizer state,
the training step, return-normalization and continuation-prevalence state,
best-evaluation state, the RSSM and continuation-head architectures, optimizer
contract, base rates, and linear-warmup length, advantage and free-bits
semantics, two-hot support, online-versus-slow value targeting, continuation
balancing, replay-sequence semantics, and MLflow run identity:

```bash
uv run --extra cpu dreamer-train \
  checkpoint_path=runs/example/checkpoints/checkpoint_final.pt \
  train.max_train_steps=20000
```

Current checkpoints embed their runtime configuration. Historical checkpoints
fall back to the adjacent `config.json`. If neither snapshot exists, resume
infers the RSSM and continuation-head architectures from model weights and uses
the historical legacy optimizer contract with no learning-rate ramp, batch
advantage normalization, straight-through free bits, 255 symmetric bins over
`[-20, 20]`, the slow-critic target, unbalanced continuation loss, and
episode-contained replay semantics. Set `allow_resume_semantic_migration=true`
to intentionally use all current authored model, optimization, loss,
value-support, target, and replay settings instead; authored architecture
changes may be incompatible with the checkpoint weights. Resume does not
restore replay contents or random-number-generator state.

### Hyperparameter Sweeps

Preregister a hypothesis, primary metric, seed set, sample budget, and stop rule
before launching a sweep. The reference optimizer contract couples all three
learning rates, so an actor-only rate sweep requires an intentional switch to
the historical `legacy` contract.

```bash
# Example bounded two-run canary under the reference contract
uv run --extra cpu dreamer-train --multirun \
  general.seed=0 train.actor_entropy_coef=3e-4,1e-3
```

### Evaluation and Visualization

```bash
# Deterministic checkpoint evaluation with diagnostics
uv run --extra cpu dreamer-inspect \
  runs/example/checkpoints/checkpoint_best.pt \
  --episodes 20 --policy_mode argmax

# Add rollout and side-by-side debug videos
uv run --extra cpu dreamer-inspect \
  runs/example/checkpoints/checkpoint_best.pt \
  --episodes 5 --policy_mode argmax \
  --save_video --compose_debug_video
```

Inspection loads the embedded configuration or adjacent `config.json`
automatically. Use `--config` only to select a fallback preset for a legacy
checkpoint that has neither.

### Monitoring

View training metrics with MLflow UI:

```bash
uv run --extra cpu mlflow ui --backend-store-uri "file://$(pwd)/runs/mlruns"
```

Then open http://localhost:5000 in your browser.

Each non-dry training run writes a versioned `run_manifest.json` beside its
configuration. The manifest records the full Git revision and dirty state,
runtime versions, configuration hash, progress, stop reason, and hashes for the
best and final checkpoints. `checkpoint_best.pt` is selected by deterministic
evaluation reward by default; `checkpoint_final.pt` is only the last state.
Every periodic evaluation in a run reuses the same reset cohort, starting at
seed `1_000_000 + general.seed`, so checkpoint comparisons are not confounded
by changing evaluation episodes.

## Configuration

Configuration is managed via Hydra with layered YAML files found in `src/dreamer/conf/`.
Hydra YAML is the authored source; `Config` is its typed runtime projection.
Invalid configurations are rejected before MLflow or training subprocesses
start.

### Run Evidence Registry

Index local manifests and legacy config snapshots without loading checkpoint
contents or deleting raw artifacts:

```bash
uv run --extra cpu python scripts/index_runs.py
```

This writes `reports/runs.csv` and `reports/runs.md`. Exact comparison keys are
only assigned to completed, clean, manifest-backed runs with a complete
evaluation protocol. Legacy runs remain visible but are marked for review.

### CartPole Research Diagnostics

The scripts below are bounded diagnostic probes, not alternate training entry
points. Run any script with `--help` for its arguments.

- `evaluate_cartpole_checkpoints.py` evaluates checkpoints on one fixed reset
  cohort; `probe_cartpole_checkpoint_drift.py` compares several checkpoints on
  histories fixed by one source policy.
- `probe_cartpole_continuation.py` audits posterior/prior continuation, while
  `probe_cartpole_continuation_supervision.py` tests whether frozen latents make
  physical failure classifiable.
- `probe_cartpole_critic_supervision.py`, `probe_cartpole_q.py`,
  `probe_cartpole_on_policy.py`, `probe_cartpole_policy_improvement.py`, and
  `probe_cartpole_rollout_fidelity.py` isolate value, action-ordering, policy,
  and matched-rollout errors. `probe_cartpole_actor_supervision.py` supplies the
  related frozen-actor baseline.
- `summarize_cartpole_replay_coverage.py` and
  `summarize_cartpole_gradient_alignment.py` reduce MLflow telemetry;
  `profile_cartpole_capacity.py` measures bounded concurrent-run resource use.

For example:

```bash
uv run --extra cpu python scripts/probe_cartpole_continuation.py --help
```

### Key Parameters

Defaults below are for the composed CartPole training configuration. The flat
`Config` defaults for the RSSM core, continuation-head depth, optimizer
contract and warmup, critic targeting, and replay sequence mode intentionally
preserve historical checkpoints and are not the authored defaults for new
runs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.max_train_steps` | 30000 | Total training updates |
| `train.batch_size` | 32 | Batch size |
| `train.sequence_length` | 32 | Sequence length for training |
| `train.eval_metric` | episode_reward | Metric used to preserve the best checkpoint |
| `train.wm_lr` / `train.actor_lr` / `train.critic_lr` | 4e-5 / 4e-5 / 4e-5 | Equal LaProp rates required by the reference optimizer contract |
| `train.gamma` / `train.horizon` / `train.contdisc` | 0.997 / 333 / true | Train continuation with `1 - 1 / horizon`; when enabled, `gamma` must match that value and is not applied again in imagination |
| `train.replay_ratio` | 1.0 | Replayed non-burn-in transitions per raw environment frame |
| `train.replay_sequence_mode` | stream | Sample same-collector, gap-free streams that may cross episode resets; `episode` preserves historical contained-window sampling |
| `train.optimizer_contract` | reference | Use equal-rate, shared-warmup optimization and let replay value loss shape observed features; `legacy` preserves split-rate detached updates |
| `train.optimizer_warmup_steps` | 1000 | Linearly ramp every optimizer's learning rate from zero; all modules still train from the first update |
| `train.critic_slow_target` | false | Use the online value for lambda returns and the actor baseline; keep the slow value as a regularizer |
| `train.critic_real_return_scale` | 0.0 | Optional full-episode replay return-to-go critic loss scale |
| `train.normalize_advantages` | false | Use only the running return-percentile scale; `true` additionally z-scores each imagined batch |
| `train.balance_continuation` | false | Preserve natural continuation probabilities; `true` applies adaptive class-balanced supervision whose raw sigmoid is not a calibrated discount |
| `train.continuation_balance_rate` | 0.01 | Terminal-prevalence EMA rate used only when continuation balancing is enabled |
| `train.free_bits_straight_through` | false | Opt into KL gradients below the one-nat free-bits threshold |
| `train.b_start` / `train.b_end` / `train.num_bins` | -20 / 20 / 255 | Symmetric symlog two-hot support shared by reward and value heads |
| `models.d_hidden` | 64 | Hidden dimension |
| `models.rssm_core` | reference | Grouped, normalized recurrent core; `legacy` preserves historical checkpoint equations and layout |
| `models.continue_head_layers` | 1 | One RMSNorm/SiLU hidden layer in the continuation head; `0` selects the legacy linear head |
| `general.use_pixels` | false | Use pixel observations |
| `general.research_gradient_diagnostics` | false | Read-only replay/WM gradient-alignment telemetry on scalar-log updates |

## Project Structure

```
src/dreamer/
├── main.py              # Training entry point (dreamer-train)
├── config.py            # Typed runtime configuration
├── inspect.py           # Checkpoint evaluation and diagnostics
├── conf/                # Hydra default configs
├── runtime/
│   ├── collector.py     # Data collection process
│   ├── env.py           # Gym environment construction
│   └── replay_buffer.py # Episode storage and sampling
├── models/              # World Model & Actor-Critic
│   ├── encoder.py       # Observation encoders
│   ├── decoder.py       # Observation decoders
│   ├── world_model.py   # RSSM dynamics model
│   └── ...
└── trainer/             # Training loop, checkpoints, logging
```

## Key Implementation Details

### DreamerV3 Features Implemented

- **Symlog encoding** for unbounded value predictions
- **Twohot discrete distributions** for reward/value prediction
- **One-nat hard free bits** by default, with an opt-in straight-through mode
- **Grouped, normalized RSSM core** with legacy recurrent-checkpoint support
- **EMA-based return normalization** for stable policy gradients
- **Online value targets** for lambda returns and the policy baseline, with the
  slow value retained as a distributional regularizer
- **Reference optimizer contract** with equal-rate linear warmup and replay-value
  representation gradients, plus legacy resume semantics

### Training Loop

1. Collector process gathers experience from the environment
2. Trainer samples batches from the replay buffer
3. World model learns to predict observations, rewards, and continuations
4. Actor-critic trains on imagined trajectories (dream rollouts)
5. Updated policy weights are synced back to the collector

## Reviewer Notes

This repository is intended to show ML engineering depth: paper-to-code
translation, modular model components, config-driven training, process-separated
collection/training, experiment logging, and tests around the highest-risk math
and runtime pieces.

What is straightforward to inspect:

- `src/dreamer/models/`: RSSM world model, encoder/decoder, losses, dreaming,
  optimizer helpers, and DreamerV3 math utilities.
- `src/dreamer/trainer/`: training loop, checkpointing, MLflow logging, and
  forward-pass helpers.
- `src/dreamer/runtime/`: environment construction, collector process, and
  replay buffer.
- `tests/`: unit/component tests for config behavior, simple custom
  environments, two-hot encoding, pixel-only world-model loss, free-bits
  gradients, actor advantage normalization, and decoder edge cases.
- `.github/workflows/pr-checks.yml`: baseline pull-request checks with stable
  `lint` and `test` jobs.

Current limitations:

- Full training is intentionally not run in hosted CI; CI verifies syntax,
  scoped reliability types, and fast tests only.
- Current retained CartPole runs do not demonstrate stable solved performance.
  Cross-reset stream replay fixes continuation-risk ordering but still collapses
  from a best return of 500 to 205. The latest audit localizes the next bounded
  canary to the coherent reference optimizer time scale and replay-value
  representation gradient.
- The README still needs a reproduced experiment report with config, seeds,
  runtime, return curve, checkpoint hash, and evaluation video/GIF.
- Generated checkpoints, MLflow runs, and videos should stay out of normal Git
  history and be linked as release artifacts or external experiment artifacts.
- The next portfolio polish pass should add one short result section with a
  plot and one trained-policy clip.

## Troubleshooting

### Common Issues

**Out of memory**: Reduce `train.batch_size` or `models.d_hidden`

**Slow training**:
- Disable `general.use_pixels` for simple envs
- Ensure GPU is being utilized (check with `nvidia-smi` or `rocm-smi`)

**Policy doesn't learn**:
- Inspect terminal/live continuation calibration and replay terminal coverage,
  then actor entropy and on-policy action-value agreement
- Under the reference optimizer contract, change world-model, actor, and critic
  rates together in a preregistered bounded canary; use `legacy` explicitly
  when investigating split rates

## References

- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
- [DreamerV3 Official Implementation](https://github.com/danijar/dreamerv3)

## License

No license file is currently included. Add an explicit license before
redistributing the project.
