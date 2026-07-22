#!/usr/bin/env python3
"""
Hydra-based training entry point for DreamerV3.

Usage:
    # Single run with defaults
    uv run dreamer-train

    # Override parameters
    uv run dreamer-train train.actor_entropy_coef=1e-3 models.d_hidden=128

    # Use environment-specific base config
    uv run dreamer-train env=cartpole_state_only
"""

import os
import random
import tempfile
from dataclasses import asdict, replace
from typing import Protocol, Sequence
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import mlflow
import numpy as np
import torch
import multiprocessing as mp

from dreamer.config import (
    Config,
    dump_config_json,
    load_checkpoint_config,
    validate_config,
)
from dreamer.run_manifest import create_run_manifest, finish_run_manifest
from dreamer.trainer import train_world_model
from dreamer.runtime.collector import collect_experiences


class ChildProcessHandle(Protocol):
    """Process operations needed by the parent shutdown contract."""

    @property
    def exitcode(self) -> int | None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...


class ProcessEvent(Protocol):
    """Event operations used by the parent/child shutdown handshake."""

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class ChildProcessError(RuntimeError):
    """Raised when a required training subprocess fails or cannot stop."""


def resolve_device(device_str: str) -> str:
    """Resolve device from config, handling 'auto' setting."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def seed_everything(seed: int) -> None:
    """Seed process-local RNGs used by training orchestration."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def join_training_processes(
    trainer: ChildProcessHandle,
    collectors: Sequence[ChildProcessHandle],
    stop_event: ProcessEvent,
    training_done_event: ProcessEvent,
    collectors_stopped_event: ProcessEvent,
    collector_timeout: float = 5.0,
    trainer_timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> None:
    """Coordinate normal shutdown without outliving trainer-owned model data.

    The parent command is the public reliability boundary. A child process that
    crashes or ignores shutdown must therefore make the command exit nonzero.
    """
    failures: list[str] = []
    premature_collectors: set[int] = set()
    training_completed = False

    while True:
        if training_done_event.wait(timeout=poll_interval):
            training_completed = True
            break

        if trainer.exitcode is not None:
            if trainer.exitcode == 0:
                failures.append("trainer exited before signaling training completion")
            else:
                failures.append(f"trainer exited with code {trainer.exitcode}")
            break

        for index, collector in enumerate(collectors):
            if collector.exitcode is not None:
                premature_collectors.add(index)
                failures.append(
                    f"collector[{index}] exited with code {collector.exitcode} "
                    "before training completed"
                )
                break
        if failures:
            break

    stop_event.set()
    for index, collector in enumerate(collectors):
        collector.join(timeout=collector_timeout)
        if collector.is_alive():
            collector.terminate()
            collector.join(timeout=collector_timeout)
            failures.append(
                f"collector[{index}] did not stop within {collector_timeout:.1f}s"
            )
        elif collector.exitcode != 0 and index not in premature_collectors:
            failures.append(
                f"collector[{index}] exited with code {collector.exitcode}"
            )

    # The trainer may still own shared-memory tensor handles referenced by a
    # collector queue. Release it only after no collector can read them.
    collectors_stopped_event.set()

    if not training_completed and trainer.is_alive():
        trainer.terminate()
    trainer.join(timeout=trainer_timeout)
    if trainer.is_alive():
        trainer.terminate()
        trainer.join(timeout=trainer_timeout)
        failures.append(
            f"trainer did not stop within {trainer_timeout:.1f}s after collectors"
        )
    elif training_completed and trainer.exitcode != 0:
        failures.append(f"trainer exited with code {trainer.exitcode}")

    if failures:
        raise ChildProcessError("; ".join(failures))


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
        if "rssm_core" in models:
            d["rssm_core"] = models.rssm_core
        if "continue_head_layers" in models:
            d["continue_head_layers"] = models.continue_head_layers
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
    unexpected = sorted(set(d) - valid_keys)
    if unexpected:
        raise ValueError(f"Unknown runtime configuration keys: {unexpected}")
    missing = sorted(valid_keys - set(d))
    if missing:
        raise ValueError(
            "Hydra config is missing runtime fields; YAML must be canonical: "
            f"{missing}"
        )

    return Config(**d)


def resolve_resume_config(
    flat_cfg: Config,
    checkpoint_path: str | Path,
    *,
    checkpoint: dict | None = None,
    allow_semantic_migration: bool = False,
) -> Config:
    """Restore compatibility-sensitive architecture and training semantics."""
    if allow_semantic_migration:
        return flat_cfg

    checkpoint_path = Path(checkpoint_path)
    if checkpoint is None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    if checkpoint is None:
        raise ValueError("checkpoint payload is empty")
    checkpoint_config = load_checkpoint_config(checkpoint_path, checkpoint)
    if checkpoint_config is not None:
        return replace(
            flat_cfg,
            rssm_core=checkpoint_config.rssm_core,
            continue_head_layers=checkpoint_config.continue_head_layers,
            critic_slow_target=checkpoint_config.critic_slow_target,
            optimizer_contract=checkpoint_config.optimizer_contract,
            optimizer_warmup_steps=checkpoint_config.optimizer_warmup_steps,
            wm_lr=checkpoint_config.wm_lr,
            actor_lr=checkpoint_config.actor_lr,
            critic_lr=checkpoint_config.critic_lr,
            normalize_advantages=checkpoint_config.normalize_advantages,
            free_bits_straight_through=(
                checkpoint_config.free_bits_straight_through
            ),
            b_start=checkpoint_config.b_start,
            b_end=checkpoint_config.b_end,
            num_bins=checkpoint_config.num_bins,
            balance_continuation=checkpoint_config.balance_continuation,
            continuation_balance_rate=checkpoint_config.continuation_balance_rate,
            replay_sequence_mode=checkpoint_config.replay_sequence_mode,
        )

    world_model_state = checkpoint.get("world_model", {})
    if "dynin_deter.0.weight" in world_model_state:
        rssm_core = "reference"
    elif "_W_ir" in world_model_state:
        rssm_core = "legacy"
    else:
        raise ValueError("checkpoint does not identify its RSSM core architecture")
    if "continue_predictor.weight" in world_model_state:
        continue_head_layers = 0
    elif "continue_predictor.0.weight" in world_model_state:
        continue_head_layers = 1
    else:
        raise ValueError(
            "checkpoint does not identify its continuation-head architecture"
        )
    return replace(
        flat_cfg,
        rssm_core=rssm_core,
        continue_head_layers=continue_head_layers,
        critic_slow_target=True,
        optimizer_contract="legacy",
        optimizer_warmup_steps=0,
        normalize_advantages=True,
        free_bits_straight_through=True,
        b_start=-20,
        b_end=20,
        num_bins=255,
        balance_continuation=False,
        replay_sequence_mode="episode",
    )


def run_training(
    flat_cfg: Config,
    mlflow_run_name: str | None = None,
    checkpoint_path: str | None = None,
    allow_resume_semantic_migration: bool = False,
):
    """Run training while preserving checkpoint semantics by default."""
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        flat_cfg = resolve_resume_config(
            flat_cfg,
            checkpoint_path,
            checkpoint=checkpoint,
            allow_semantic_migration=allow_resume_semantic_migration,
        )
    validate_config(flat_cfg)
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
    print(f"  seed: {flat_cfg.seed}")
    print(f"  Output: {log_dir}")

    seed_everything(flat_cfg.seed)

    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")

    # Dump config JSON for repro
    config_path = log_dir / "config.json"
    dump_config_json(flat_cfg, str(config_path))
    print(f"  Config saved: {config_path}")

    # Setup MLflow
    mlflow_run_id = None
    checkpoint_mlflow_run_id = None
    obs_mode = (
        "hybrid"
        if flat_cfg.use_pixels and flat_cfg.n_observations > 0
        else "vision"
        if flat_cfg.use_pixels
        else "vector"
    )
    if checkpoint is not None:
        try:
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

        if checkpoint_mlflow_run_id:
            mlflow.set_tag("resumed_from_checkpoint", checkpoint_path)
        else:
            # Log config as params only for new runs.
            for k, v in asdict(flat_cfg).items():
                if isinstance(v, str) and len(v) > 250:
                    v = v[:250] + "..."
                mlflow.log_param(k, str(v))

    manifest = create_run_manifest(
        log_dir=log_dir,
        config=flat_cfg,
        device=device,
        mlflow_run_id=mlflow_run_id,
        checkpoint_path=checkpoint_path,
    )
    if not flat_cfg.dry_run:
        source = manifest["source"]
        if source["commit"]:
            mlflow.set_tag("git_commit", source["commit"])
        if source["branch"]:
            mlflow.set_tag("git_branch", source["branch"])
        if source["dirty"] is not None:
            mlflow.set_tag("git_dirty", str(source["dirty"]).lower())
        mlflow.set_tag("run_manifest_id", manifest["run_id"])
        mlflow.set_tag("obs_mode", obs_mode)

    try:
        mp_ctx = mp.get_context("spawn")
        # Queue stores full episodes, so tying maxsize to batch_size can over-buffer
        # large pixel episodes and trigger host OOM. Keep this bounded by collectors.
        queue_max_episodes = min(16, max(4, flat_cfg.num_collectors * 4))
        data_queue = mp_ctx.Queue(maxsize=queue_max_episodes)
        collector_count = flat_cfg.num_collectors
        model_queues = [mp_ctx.Queue(maxsize=1) for _ in range(collector_count)]
        stop_event = mp_ctx.Event()
        training_done_event = mp_ctx.Event()
        collectors_stopped_event = mp_ctx.Event()

        experience_loops = []
        for collector_id, model_queue in enumerate(model_queues):
            p = mp_ctx.Process(
                target=collect_experiences,
                args=(
                    data_queue,
                    model_queue,
                    flat_cfg,
                    stop_event,
                    str(log_dir),
                    checkpoint_path,
                    collector_id,
                ),
            )
            experience_loops.append(p)
        trainer_loop = mp_ctx.Process(
            target=train_world_model,
            args=(
                flat_cfg,
                data_queue,
                model_queues,
                training_done_event,
                collectors_stopped_event,
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

        join_training_processes(
            trainer_loop,
            experience_loops,
            stop_event,
            training_done_event,
            collectors_stopped_event,
        )
        finish_run_manifest(log_dir, status="completed")
        print("Training complete.")
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        try:
            finish_run_manifest(
                log_dir,
                status=status,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as manifest_exc:
            # Preserve the training failure as the command's primary error. A
            # secondary metadata write failure is still useful diagnostic data.
            print(f"  Warning: could not finalize run manifest: {manifest_exc}")
        raise
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
    allow_resume_semantic_migration = bool(
        cfg.get("allow_resume_semantic_migration", False)
    )

    from hydra.core.hydra_config import HydraConfig

    run_name = Path(HydraConfig.get().runtime.output_dir).name
    run_training(
        flat_cfg,
        mlflow_run_name=run_name,
        checkpoint_path=checkpoint_path,
        allow_resume_semantic_migration=allow_resume_semantic_migration,
    )


if __name__ == "__main__":
    main()
