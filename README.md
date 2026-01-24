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
git clone https://github.com/yourusername/SleepyDreamyV3.git
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
uv run dreamer-train train.actor_lr=3e-5 train.max_steps=50000

# Use a different environment config
uv run dreamer-train +env=cartpole_vision

# Smoke test / dry run (no MLflow, no checkpoints, temp directory)
uv run dreamer-train general.dry_run=true train.max_steps=100
```

### Hyperparameter Sweeps

```bash
# Grid search over learning rates
uv run dreamer-train --multirun train.actor_lr=1e-5,3e-5,1e-4

# Use predefined sweep config
uv run dreamer-train --multirun +sweep=ac_params
```

### Evaluation

```bash
# Evaluate a trained checkpoint
uv run dreamer-eval --checkpoint runs/01-17_1234/checkpoints/checkpoint_final.pt
```

### Visualization

```bash
# Generate dream videos from a checkpoint
uv run dreamer-viz --checkpoint runs/01-17_1234/checkpoints/checkpoint_final.pt
```

### Monitoring

View training metrics with MLflow UI:

```bash
mlflow ui --backend-store-uri file://runs/mlruns
```

Then open http://localhost:5000 in your browser.

## Configuration

Configuration is managed via Hydra with layered YAML files found in `src/dreamer/conf/`.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.max_steps` | 30000 | Total training steps |
| `train.batch_size` | 32 | Batch size |
| `train.sequence_length` | 25 | Sequence length for training |
| `models.d_hidden` | 64 | Hidden dimension |
| `general.use_pixels` | false | Use pixel observations |

## Project Structure

```
src/dreamer/
├── main.py              # Training entry point (dreamer-train)
├── conf/                # Hydra default configs
├── data/
│   └── replay_buffer.py # Episode storage
├── envs/
│   ├── collector.py     # Data collection process
│   └── utils.py         # Gym environment creation
├── models/              # World Model & Actor-Critic
│   ├── encoder.py       # Observation encoders
│   ├── decoder.py       # Observation decoders
│   ├── world_model.py   # RSSM dynamics model
│   └── ...
├── scripts/             # Utility scripts
│   ├── evaluate.py      # dreamer-eval
│   ├── visualize.py     # dreamer-viz
```

## Training Modes

1. **`train`** (default): Full training with warmup period where only the world model trains before actor-critic

2. **`bootstrap`**: Train only the world model using random actions. Useful for pre-training dynamics.

3. **`dreamer`**: Train actor-critic with a frozen pre-trained world model. Requires `--checkpoint`.

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

MIT
