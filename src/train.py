#!/usr/bin/env python3
"""
Hydra-based training entry point for DreamerV3.

Usage:
    # Single run with defaults
    uv run python -m src.train

    # Override parameters
    uv run python -m src.train train.actor_lr=3e-5 train.actor_entropy_coef=0.01

    # Use different environment config
    uv run python -m src.train +env=cartpole_vision

    # Multirun sweep (parallel)
    uv run python -m src.train --multirun train.actor_lr=1e-5,3e-5,1e-4

    # Resume from checkpoint
    uv run python -m src.train checkpoint=/path/to/checkpoint.pt

    # Dreamer mode (AC training with fixed WM)
    uv run python -m src.train mode=dreamer checkpoint=/path/to/wm.pt
"""

import os

# Set AMD ROCm env var before any torch imports
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp

from .config import adapt_config
from .trainer import train_world_model
from .environment import collect_experiences


def run_training(cfg: DictConfig, mode: str = "train", checkpoint_path: str | None = None):
    """Run training with the given configuration."""
    # Adapt Hydra config for backward compatibility
    config = adapt_config(cfg)

    # Get output directory from Hydra
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print(f"Training Configuration:")
    print(f"  Environment: {config.environment.environment_name}")
    print(f"  Device: {config.general.device}")
    print(f"  d_hidden: {config.models.d_hidden}")
    print(f"  batch_size: {config.train.batch_size}")
    print(f"  actor_lr: {config.train.actor_lr}")
    print(f"  actor_entropy_coef: {config.train.actor_entropy_coef}")
    print(f"  Output: {log_dir}")

    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")

    # Create queues for inter-process communication
    data_queue = mp.Queue(maxsize=config.train.batch_size * 5)
    model_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    # Determine reset_ac based on mode
    reset_ac = mode == "dreamer"

    # Launch processes
    experience_loop = mp.Process(
        target=collect_experiences,
        args=(data_queue, model_queue, config, stop_event, log_dir),
    )
    trainer_loop = mp.Process(
        target=train_world_model,
        args=(
            config,
            data_queue,
            model_queue,
            log_dir,
            checkpoint_path,
            mode,
            reset_ac,
        ),
    )

    experience_loop.start()
    trainer_loop.start()

    trainer_loop.join()
    stop_event.set()
    experience_loop.join(timeout=5.0)
    if experience_loop.is_alive():
        experience_loop.terminate()

    print("Training complete.")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry point."""
    # Required for CUDA with multiprocessing
    mp.set_start_method("spawn", force=True)

    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    # Determine mode and checkpoint from config overrides
    mode = cfg.get("mode", "train")
    checkpoint = cfg.get("checkpoint", None)

    run_training(cfg, mode=mode, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
