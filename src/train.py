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

    # Multirun sweep (sequential)
    uv run python -m src.train --multirun train.actor_lr=1e-5,3e-5,1e-4
    # Or use predefined sweep config
    uv run python -m src.train --multirun +sweep=ac_params

    # Resume from checkpoint
    uv run python -m src.train checkpoint=/path/to/checkpoint.pt

    # Dreamer mode (AC training with fixed WM)
    uv run python -m src.train mode=dreamer checkpoint=/path/to/wm.pt

    # Dry run (smoke test - no MLflow, no checkpoints, temp directory)
    uv run python -m src.train general.dry_run=true train.max_steps=100
"""

import os
import subprocess
import socket

# Set AMD ROCm env var before any torch imports
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_sweep_info() -> dict:
    """
    Extract sweep information from Hydra config.

    Returns:
        dict with keys:
        - is_sweep: bool
        - sweep_id: str | None (sweep directory basename, e.g., "01-17_1430")
        - varied_params: dict (parsed overrides, e.g., {"actor_lr": "1e-4"})
    """
    try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        is_sweep = hydra_cfg.mode.name == "MULTIRUN"

        if is_sweep:
            sweep_dir = hydra_cfg.sweep.dir
            sweep_id = os.path.basename(sweep_dir)
            override_dirname = hydra_cfg.job.override_dirname

            # Parse override_dirname to extract varied params
            # Format: "train.actor_lr=1e-4,train.gamma=0.99"
            varied_params = {}
            if override_dirname:
                for part in override_dirname.split(","):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        # Use short key (last part after dot)
                        short_key = key.split(".")[-1]
                        varied_params[short_key] = value

            return {
                "is_sweep": True,
                "sweep_id": sweep_id,
                "varied_params": varied_params,
            }
    except Exception:
        pass

    return {"is_sweep": False, "sweep_id": None, "varied_params": {}}


def generate_run_name(log_dir: str, sweep_info: dict) -> str:
    """
    Generate a descriptive run name.

    Format:
    - Single run: "01-17_1430" (timestamp from Hydra)
    - Sweep run: "lr=1e-4_gamma=0.99" (varied params only, sweep_id in tags)
    """
    base_name = os.path.basename(log_dir)

    if sweep_info["is_sweep"] and sweep_info["varied_params"]:
        # For sweeps, show the varied params (makes comparing runs easy)
        return "_".join(f"{k}={v}" for k, v in sweep_info["varied_params"].items())

    return base_name


def start_mlflow_ui(mlruns_dir: str, port: int = 5000) -> subprocess.Popen | None:
    """Start MLflow UI server in background if not already running."""
    if is_port_in_use(port):
        print(f"  MLflow UI already running at http://localhost:{port}")
        return None

    proc = subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", f"file://{mlruns_dir}", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  MLflow UI started at http://localhost:{port}")
    return proc

from .config import adapt_config
from .trainer import train_world_model
from .environment import collect_experiences


def flatten_config(cfg: DictConfig, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested OmegaConf config to a flat dict with dot notation keys.

    Example: {"train": {"lr": 0.001}} -> {"train.lr": 0.001}
    """
    items = {}
    for key, value in cfg.items():
        key_str = str(key)
        new_key = f"{parent_key}{sep}{key_str}" if parent_key else key_str
        if isinstance(value, DictConfig):
            items.update(flatten_config(value, new_key, sep))
        else:
            # Convert to basic Python types for MLflow
            if hasattr(value, "item"):  # numpy/torch scalar
                value = value.item()
            items[new_key] = value
    return items


def run_training(cfg: DictConfig, mode: str = "train", checkpoint_path: str | None = None):
    """Run training with the given configuration."""
    import tempfile

    # Adapt Hydra config for backward compatibility
    config = adapt_config(cfg)

    # Check for dry_run mode (smoke tests - no MLflow, no checkpoints)
    dry_run = cfg.get("general", {}).get("dry_run", False)

    # Get output directory from Hydra (or temp dir for dry_run)
    if dry_run:
        log_dir = tempfile.mkdtemp(prefix="dreamer_dry_")
        print(f"DRY RUN MODE - no MLflow, no checkpoints, temp dir: {log_dir}")
    else:
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

    # Setup MLflow tracking (skip in dry_run mode)
    mlflow_run_id = None
    is_resumed_run = False

    if not dry_run:
        mlruns_dir = os.path.join(os.path.dirname(log_dir), "mlruns")
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        mlflow.set_experiment(f"DreamerV3-{config.environment.environment_name}")

        # Check for existing run_id from checkpoint for resume support
        if checkpoint_path:
            import torch
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            mlflow_run_id = ckpt.get("mlflow_run_id")
            if mlflow_run_id:
                print(f"  Resuming MLflow run: {mlflow_run_id}")
                is_resumed_run = True

        # Get sweep info and generate descriptive run name
        sweep_info = get_sweep_info()
        run_name = generate_run_name(log_dir, sweep_info)

        # Start MLflow run (resume if run_id exists)
        run = mlflow.start_run(run_id=mlflow_run_id, run_name=run_name)
        mlflow_run_id = run.info.run_id
        print(f"  MLflow run ID: {mlflow_run_id}")
        print(f"  MLflow tracking: {mlruns_dir}")

        # Add useful tags for filtering/grouping
        git_commit = get_git_commit()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)

        if sweep_info["is_sweep"]:
            mlflow.set_tag("sweep_id", sweep_info["sweep_id"])
            mlflow.set_tag("run_type", "sweep")
            print(f"  Sweep ID: {sweep_info['sweep_id']}")
        else:
            mlflow.set_tag("run_type", "single")

        # Start MLflow UI server
        start_mlflow_ui(mlruns_dir, port=5000)

    # Log flattened config as parameters (only on new runs, skip in dry_run)
    if not dry_run and not is_resumed_run:
        flat_params = flatten_config(cfg)
        # MLflow has a 500 param limit, truncate long values
        for k, v in flat_params.items():
            if isinstance(v, str) and len(v) > 250:
                flat_params[k] = v[:250] + "..."
        mlflow.log_params(flat_params)

    try:
        # Use explicit spawn context for CUDA compatibility
        mp_ctx = mp.get_context("spawn")

        # Create queues for inter-process communication
        data_queue = mp_ctx.Queue(maxsize=config.train.batch_size * 5)
        model_queue = mp_ctx.Queue(maxsize=1)
        stop_event = mp_ctx.Event()

        # Determine reset_ac based on mode
        reset_ac = mode == "dreamer"

        # Launch processes
        experience_loop = mp_ctx.Process(
            target=collect_experiences,
            args=(data_queue, model_queue, config, stop_event, log_dir),
        )
        trainer_loop = mp_ctx.Process(
            target=train_world_model,
            args=(
                config,
                data_queue,
                model_queue,
                log_dir,
                checkpoint_path,
                mode,
                reset_ac,
                mlflow_run_id,  # Pass MLflow run ID to trainer (None in dry_run)
                dry_run,
            ),
        )

        experience_loop.start()
        trainer_loop.start()

        trainer_loop.join()
        if trainer_loop.exitcode not in (0, None):
            print(f"Trainer exited with code {trainer_loop.exitcode}")
        stop_event.set()
        experience_loop.join(timeout=5.0)
        if experience_loop.is_alive():
            experience_loop.terminate()
        if experience_loop.exitcode not in (0, None):
            print(f"Collector exited with code {experience_loop.exitcode}")

        print("Training complete.")
    finally:
        if not dry_run:
            mlflow.end_run()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry point."""
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    # Determine mode and checkpoint from config overrides
    mode = cfg.get("mode", "train")
    checkpoint = cfg.get("checkpoint", None)

    run_training(cfg, mode=mode, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
