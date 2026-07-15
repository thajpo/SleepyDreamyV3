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

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### GPU Support

- **NVIDIA CUDA**: Works out of the box with PyTorch
- **AMD ROCm**: Configured in `pyproject.toml` for ROCm 6.4 (set `HSA_OVERRIDE_GFX_VERSION` if needed)

## Quick Start

### Training

```bash
# Full training with default config (CartPole, state-only)
uv run dreamer-train

# Override parameters via CLI
uv run dreamer-train train.actor_lr=3e-5 train.max_train_steps=50000

# Smoke test / dry run (no MLflow, no checkpoints, temp directory)
uv run dreamer-train \
  general.dry_run=true general.device=cpu \
  train.max_train_steps=1 train.min_buffer_episodes=2 \
  train.batch_size=2 train.sequence_length=4 \
  train.replay_burn_in=1 train.eval_every=0
```

### Hyperparameter Sweeps

```bash
# Grid search over learning rates
uv run dreamer-train --multirun train.actor_lr=1e-5,3e-5,1e-4

# Use predefined sweep config
uv run dreamer-train --multirun +sweep=ac_params
```

### Evaluation and Visualization

```bash
# Deterministic checkpoint evaluation with diagnostics
uv run dreamer-inspect \
  runs/example/checkpoints/checkpoint_final.pt \
  --episodes 20 --policy_mode argmax

# Add rollout and side-by-side debug videos
uv run dreamer-inspect \
  runs/example/checkpoints/checkpoint_final.pt \
  --episodes 5 --policy_mode argmax \
  --save_video --compose_debug_video
```

### Monitoring

View training metrics with MLflow UI:

```bash
uv run mlflow ui --backend-store-uri "file://$(pwd)/runs/mlruns"
```

Then open http://localhost:5000 in your browser.

## Configuration

Configuration is managed via Hydra with layered YAML files found in `src/dreamer/conf/`.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.max_train_steps` | 30000 | Total training updates |
| `train.batch_size` | 32 | Batch size |
| `train.sequence_length` | 25 | Sequence length for training |
| `models.d_hidden` | 64 | Hidden dimension |
| `general.use_pixels` | false | Use pixel observations |

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
- **Free bits** with straight-through gradients for KL regularization
- **Block-diagonal GRU** for efficient recurrent state updates
- **EMA-based return normalization** for stable policy gradients

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

- Full training is intentionally not run in hosted CI; CI verifies syntax and
  fast tests only.
- Current retained CartPole runs do not provide a reproduced solved checkpoint.
  The latest research notes localize the remaining problem to policy improvement:
  learned latents contain useful control information, but the actor often
  collapses to a constant action or remains random-like.
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
- Increase `train.actor_warmup_steps` to give the world model more time
- Try different `train.actor_lr` values (sweep recommended)

## References

- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
- [DreamerV3 Official Implementation](https://github.com/danijar/dreamerv3)

## License

No license file is currently included. Add an explicit license before
redistributing the project.
