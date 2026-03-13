#!/usr/bin/env python3
"""
Hydra-based training entry point for DreamerV3.

Usage:
    # Single run with defaults
    uv run dreamer-train

    # Override parameters
    uv run dreamer-train train.wm_lr=5e-4 models.d_hidden=128

    # Use environment-specific base config
    uv run dreamer-train +env=cartpole
"""

import os
import subprocess
import socket
import tempfile
from dataclasses import asdict
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import mlflow
import torch
import multiprocessing as mp

from dreamer.config import (
    Config,
    dump_config_json,
)
from dreamer.trainer import train_world_model
from dreamer.runtime.collector import collect_experiences


def resolve_device(device_str: str) -> str:
    """Resolve device from config, handling 'auto' setting."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


# move this mf outta main
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


def start_mlflow_ui(mlruns_dir: str, port: int = 5000) -> subprocess.Popen | None:
    """Start MLflow UI server in background if not already running."""
    if is_port_in_use(port):
        print(f"  MLflow UI already running at http://localhost:{port}")
        return None

    proc = subprocess.Popen(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            f"file://{mlruns_dir}",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  MLflow UI started at http://localhost:{port}")
    return proc


def dictconfig_to_config(cfg: DictConfig) -> Config:
    """Map nested Hydra config back to the flat Config dataclass."""
    d = {}
    if "general" in cfg:
        d.update(cfg.general)
    if "environment" in cfg:
        d.update(cfg.environment)
    if "models" in cfg:
        models = cfg.models
        if "d_hidden" in models:
            d["d_hidden"] = models.d_hidden
        if "num_latents" in models:
            d["num_latents"] = models.num_latents
        if "encoder" in models:
            enc = models.encoder
            if "cnn" in enc:
                for k, v in enc.cnn.items():
                    if k == "target_size":
                        v = tuple(v)
                    d[f"encoder_cnn_{k}"] = v
            if "mlp" in enc:
                for k, v in enc.mlp.items():
                    d[f"encoder_mlp_{k}"] = v
        if "rnn" in models and "n_blocks" in models.rnn:
            d["rnn_n_blocks"] = models.rnn.n_blocks
    if "train" in cfg:
        d.update(cfg.train)

    import inspect

    valid_keys = set(inspect.signature(Config).parameters.keys())
    filtered_d = {k: v for k, v in d.items() if k in valid_keys}

    return Config(**filtered_d)


def run_training(
    flat_cfg: Config,
    mlflow_run_name: str | None = None,
    checkpoint_path: str | None = None,
):
    """Run training with the given configuration."""
    device = resolve_device(flat_cfg.device)

    if flat_cfg.dry_run:
        log_dir = Path(tempfile.mkdtemp(prefix="dreamer_dry_"))
        print(f"DRY RUN MODE - no MLflow, no checkpoints, temp dir: {log_dir}")
    else:
        from hydra.core.hydra_config import HydraConfig

        log_dir = Path(HydraConfig.get().runtime.output_dir)

    print(f"Training Configuration:")
    print(f"  Environment: {flat_cfg.environment_name}")
    print(f"  Device: {device}")
    print(f"  d_hidden: {flat_cfg.d_hidden}")
    print(f"  batch_size: {flat_cfg.batch_size}")
    print(f"  actor_lr: {flat_cfg.actor_lr}")
    print(f"  actor_entropy_coef: {flat_cfg.actor_entropy_coef}")
    print(f"  Output: {log_dir}")

    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")

    # Dump config JSON for repro
    config_path = log_dir / "config.json"
    dump_config_json(flat_cfg, str(config_path))
    print(f"  Config saved: {config_path}")

    # Setup MLflow
    mlflow_run_id = None
    checkpoint_mlflow_run_id = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint_mlflow_run_id = checkpoint.get("mlflow_run_id")
        except Exception as exc:
            print(f"  Warning: could not read MLflow run id from checkpoint: {exc}")

    if not flat_cfg.dry_run:
        mlruns_dir = Path("runs") / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlruns_dir.resolve()}")

        exp_name = flat_cfg.experiment_name or f"DreamerV3-{flat_cfg.environment_name}"
        mlflow.set_experiment(exp_name)

        if checkpoint_mlflow_run_id:
            run = mlflow.start_run(run_id=checkpoint_mlflow_run_id)
        else:
            run_name = (
                mlflow_run_name
                or f"{exp_name}_{datetime.now().strftime('%m-%d_%H%M%S')}"
            )
            run = mlflow.start_run(run_name=run_name)
        mlflow_run_id = run.info.run_id
        print(f"  MLflow run ID: {mlflow_run_id}")
        print(f"  MLflow tracking: {mlruns_dir}")

        # Tags
        git_commit = get_git_commit()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)
        obs_mode = (
            "hybrid"
            if flat_cfg.use_pixels and flat_cfg.n_observations > 0
            else "vision"
            if flat_cfg.use_pixels
            else "vector"
        )
        mlflow.set_tag("obs_mode", obs_mode)

        start_mlflow_ui(str(mlruns_dir), port=5000)

        if checkpoint_mlflow_run_id:
            mlflow.set_tag("resumed_from_checkpoint", checkpoint_path)
        else:
            # Log config as params only for new runs.
            for k, v in asdict(flat_cfg).items():
                if isinstance(v, str) and len(v) > 250:
                    v = v[:250] + "..."
                mlflow.log_param(k, str(v))

    try:
        mp_ctx = mp.get_context("spawn")
        # Queue stores full episodes, so tying maxsize to batch_size can over-buffer
        # large pixel episodes and trigger host OOM. Keep this bounded by collectors.
        queue_max_episodes = min(16, max(4, int(flat_cfg.num_collectors) * 4))
        data_queue = mp_ctx.Queue(maxsize=queue_max_episodes)
        model_queue = mp_ctx.Queue(maxsize=1)
        stop_event = mp_ctx.Event()

        experience_loops = []
        for _ in range(max(1, flat_cfg.num_collectors)):
            p = mp_ctx.Process(
                target=collect_experiences,
                args=(
                    data_queue,
                    model_queue,
                    flat_cfg,
                    stop_event,
                    str(log_dir),
                    checkpoint_path,
                ),
            )
            experience_loops.append(p)
        trainer_loop = mp_ctx.Process(
            target=train_world_model,
            args=(
                flat_cfg,
                data_queue,
                model_queue,
                str(log_dir),
                checkpoint_path,
                mlflow_run_id,
                flat_cfg.dry_run,
                device,
            ),
        )

        for p in experience_loops:
            p.start()
        trainer_loop.start()

        trainer_loop.join()
        if trainer_loop.exitcode not in (0, None):
            print(f"Trainer exited with code {trainer_loop.exitcode}")
        stop_event.set()
        for idx, p in enumerate(experience_loops):
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
            if p.exitcode not in (0, None):
                print(f"Collector[{idx}] exited with code {p.exitcode}")

        print("Training complete.")
    finally:
        if not flat_cfg.dry_run:
            mlflow.end_run()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Hydra Config ===")
    print(OmegaConf.to_yaml(cfg))
    print("====================")

    flat_cfg = dictconfig_to_config(cfg)
    checkpoint_path = cfg.get("checkpoint_path", None)

    from hydra.core.hydra_config import HydraConfig

    run_name = Path(HydraConfig.get().runtime.output_dir).name
    run_training(flat_cfg, mlflow_run_name=run_name, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
