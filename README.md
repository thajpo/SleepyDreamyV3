# SleepyDreamyV3

A PyTorch implementation of [DreamerV3](https://arxiv.org/abs/2301.04104) - a model-based reinforcement learning agent that learns world models and uses imagination for policy optimization.

## Features

- **World Model Learning**: Learns environment dynamics from experience using RSSM (Recurrent State-Space Model)
- **Imagination-Based Training**: Trains actor-critic policies entirely within the learned world model
- **State-Only & Pixel Modes**: Supports both low-dimensional state vectors and pixel observations
- **Hydra Configuration**: Flexible configuration management with YAML files and CLI overrides
- **Multiprocessing Architecture**: Separate collector and trainer processes for efficient GPU utilization
- **TensorBoard Logging**: Real-time training metrics and visualizations

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
uv run python -m src.train

# Override parameters via CLI
uv run python -m src.train train.actor_lr=3e-5 train.max_steps=50000

# Use a different environment config
uv run python -m src.train +env=cartpole_vision

# Two-phase training workflow:
# 1. Bootstrap world model (random policy)
uv run python -m src.train +mode=bootstrap train.max_steps=10000

# 2. Train actor-critic with frozen world model
uv run python -m src.train +mode=dreamer +checkpoint=runs/01-17_1234/checkpoints/checkpoint_final.pt
```

### Hyperparameter Sweeps

```bash
# Grid search over learning rates
uv run python -m src.train --multirun train.actor_lr=1e-5,3e-5,1e-4

# Use predefined sweep config
uv run python -m src.train --multirun +sweep=ac_params
```

### Evaluation

```bash
# Evaluate a trained checkpoint
uv run python -m src.evaluate --checkpoint runs/01-17_1234/checkpoints/checkpoint_final.pt
```

### Monitoring

TensorBoard starts automatically during training:

```bash
# Or start manually
tensorboard --logdir runs/
```

## Configuration

Configuration is managed via Hydra with layered YAML files:

```
conf/
├── config.yaml          # Base defaults
├── env/                  # Environment-specific configs
│   ├── cartpole_state_only.yaml
│   └── cartpole_vision.yaml
└── sweep/                # Hyperparameter sweep configs
    └── ac_params.yaml
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.max_steps` | 30000 | Total training steps |
| `train.batch_size` | 32 | Batch size |
| `train.sequence_length` | 25 | Sequence length for training |
| `train.actor_lr` | 1e-4 | Actor learning rate |
| `train.critic_lr` | 1e-4 | Critic learning rate |
| `train.wm_lr` | 1e-4 | World model learning rate |
| `train.actor_entropy_coef` | 1e-3 | Entropy regularization |
| `train.gamma` | 0.99 | Discount factor |
| `train.lam` | 0.95 | GAE lambda |
| `train.num_dream_steps` | 15 | Imagination horizon |
| `models.d_hidden` | 64 | Hidden dimension |
| `general.use_pixels` | false | Use pixel observations |

## Project Structure

```
src/
├── main.py              # Legacy argparse entry point
├── train.py             # Hydra-based entry point
├── evaluate.py          # Evaluation script
├── config.py            # Configuration dataclasses
├── environment.py       # Data collection process
├── replay_buffer.py     # Episode storage
├── encoder.py           # CNN + MLP encoders
├── decoder.py           # Observation decoders
├── world_model.py       # RSSM dynamics model
├── trainer/             # Training logic
│   ├── core.py          # Main training loop
│   ├── losses.py        # Loss computations
│   ├── dreaming.py      # Imagination rollouts
│   ├── checkpoints.py   # Save/load utilities
│   └── ...
└── utils/               # Shared utilities
    ├── config_loader.py
    └── environment.py
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
