#!/usr/bin/env python3
"""
Argparse-based training entry point for DreamerV3 (post-Hydra migration).

Usage:
    # Single run with defaults
    uv run dreamer-train

    # Override parameters
    uv run dreamer-train --wm_lr 5e-4 --d_hidden 128

    # Use environment-specific base config
    uv run dreamer-train --config cartpole

    # Dry run
    uv run dreamer-train --dry_run --max_train_steps 100
"""

import argparse
import os
import subprocess
import socket
import tempfile
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import mlflow
import torch
from dataclasses import asdict
import multiprocessing as mp

from dreamer.config import (
    Config,
    default_config,
    cartpole_config,
    ratio_sweep_5e4_config,
    paper_cartpole_config,
    atari_pong_config,
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


def _list_other_trainers() -> List[Tuple[int, str]]:
    """Return running Dreamer trainer processes excluding current/parent process.

    We intentionally block concurrent trainer launches to avoid resource contention
    and orphaned multiprocessing trees.
    """
    current_pid = os.getpid()
    parent_pid = os.getppid()
    ignore_pids = {current_pid, parent_pid}
    matches: list[tuple[int, str]] = []

    proc_root = Path("/proc")
    if not proc_root.exists():
        return matches

    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid in ignore_pids:
            continue

        cmdline_path = proc_dir / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not raw:
            continue

        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        if not cmd:
            continue

        if "dreamer-train" in cmd or "dreamer.main" in cmd:
            matches.append((pid, cmd))

    return matches


def ensure_single_trainer() -> None:
    """Fail fast if another trainer process is already active."""
    others = _list_other_trainers()
    if not others:
        return

    print("Refusing to start: another Dreamer trainer process is already running:")
    for pid, cmd in others[:5]:
        print(f"  PID {pid}: {cmd}")
    if len(others) > 5:
        print(f"  ... and {len(others) - 5} more")
    raise SystemExit(1)


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argparse parser for Config fields."""
    parser = argparse.ArgumentParser(description="Train DreamerV3")
    parser.add_argument(
        "--config",
        choices=[
            "default",
            "cartpole",
            "ratio_sweep_5e4",
            "paper_cartpole",
            "atari_pong",
        ],
        default="default",
        help="Base config to use",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode (no MLflow, no checkpoints)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint to resume from",
    )

    # Flatten all Config fields into CLI arguments
    parser.add_argument("--device", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--use_pixels", type=bool, help="Use pixel observations")
    parser.add_argument("--profile", type=bool, help="Enable PyTorch profiler")
    parser.add_argument("--compile_models", type=bool, help="Use torch.compile")
    parser.add_argument("--experiment_name", type=str, help="MLflow experiment name")
    parser.add_argument(
        "--log_profile",
        type=str,
        help="Logging profile (lean/full)",
    )

    # Environment
    parser.add_argument("--environment_name", type=str, help="Gym environment name")
    parser.add_argument("--n_actions", type=int, help="Action space size")
    parser.add_argument("--n_observations", type=int, help="Observation vector size")
    parser.add_argument(
        "--atari_compat_mode",
        type=bool,
        help="Enable Atari compatibility wrapper stack",
    )
    parser.add_argument("--atari_noop_max", type=int, help="Atari reset no-op max")
    parser.add_argument(
        "--atari_frame_skip", type=int, help="Atari frame skip in preprocessing"
    )
    parser.add_argument(
        "--atari_terminal_on_life_loss",
        type=bool,
        help="Atari terminal-on-life-loss setting",
    )
    parser.add_argument(
        "--atari_sticky_action_prob",
        type=float,
        help="Atari sticky action probability",
    )
    parser.add_argument(
        "--atari_full_action_space",
        type=bool,
        help="Use full Atari action space",
    )
    parser.add_argument(
        "--atari_fire_reset",
        type=bool,
        help="Press FIRE once on Atari reset",
    )

    # Model architecture
    parser.add_argument("--d_hidden", type=int, help="RSSM hidden size")
    parser.add_argument(
        "--num_latents", type=int, help="Number of discrete latent variables"
    )

    # Encoder CNN
    parser.add_argument("--encoder_cnn_stride", type=int, help="CNN stride")
    parser.add_argument("--encoder_cnn_kernel_size", type=int, help="CNN kernel size")
    parser.add_argument("--encoder_cnn_num_layers", type=int, help="CNN layers")
    parser.add_argument(
        "--encoder_mlp_hidden_dim_ratio", type=int, help="Encoder MLP hidden dim ratio"
    )
    parser.add_argument("--encoder_mlp_n_layers", type=int, help="Encoder MLP layers")
    parser.add_argument("--rnn_n_blocks", type=int, help="RSSM GRU block count")

    # Training core
    parser.add_argument("--max_train_steps", type=int, help="Max training steps")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--sequence_length", type=int, help="Sequence length")
    parser.add_argument("--wm_lr", type=float, help="World model learning rate")
    parser.add_argument("--actor_lr", type=float, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, help="Critic learning rate")
    parser.add_argument("--weight_decay", type=float, help="L2 weight decay")
    parser.add_argument("--critic_ema_decay", type=float, help="Critic EMA decay")
    parser.add_argument(
        "--critic_ema_regularizer", type=float, help="Critic EMA regularizer"
    )
    parser.add_argument(
        "--critic_replay_scale", type=float, help="Critic replay loss scale"
    )
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--lam", type=float, help="TD lambda")
    parser.add_argument("--num_dream_steps", type=int, help="Imagination horizon")
    parser.add_argument(
        "--actor_entropy_coef", type=float, help="Actor entropy coefficient"
    )
    parser.add_argument(
        "--normalize_advantages", type=bool, help="Normalize advantages"
    )
    parser.add_argument("--beta_dyn", type=float, help="Dynamics KL weight")
    parser.add_argument("--beta_rep", type=float, help="Representation KL weight")
    parser.add_argument("--beta_pred", type=float, help="Prediction weight")
    parser.add_argument("--b_start", type=int, help="Reward bin start")
    parser.add_argument("--b_end", type=int, help="Reward bin end")
    parser.add_argument("--wm_ac_ratio", type=int, help="WM updates per AC update")
    parser.add_argument("--lr_cosine_decay", type=bool, help="Cosine LR decay")
    parser.add_argument(
        "--lr_cosine_min_factor", type=float, help="Cosine LR min factor"
    )
    parser.add_argument(
        "--wm_ac_ratio_cosine", type=bool, help="Cosine WM:AC ratio schedule"
    )
    parser.add_argument("--wm_ac_ratio_max", type=int, help="Max WM:AC ratio")
    parser.add_argument("--wm_ac_ratio_min", type=int, help="Min WM:AC ratio")
    parser.add_argument("--wm_ac_ratio_invert", type=bool, help="Invert WM:AC schedule")
    parser.add_argument(
        "--surprise_scale_ac_lr", type=bool, help="Surprise-scaled AC LR"
    )
    parser.add_argument("--surprise_lr_scale_k", type=float, help="Surprise LR scale k")
    parser.add_argument("--surprise_ema_beta", type=float, help="Surprise EMA beta")
    parser.add_argument(
        "--surprise_wm_focus_threshold", type=float, help="WM focus threshold"
    )
    parser.add_argument("--surprise_wm_focus_ratio", type=int, help="WM focus ratio")
    parser.add_argument(
        "--surprise_wm_focus_duration", type=int, help="WM focus duration"
    )
    parser.add_argument(
        "--surprise_wm_focus_cooldown", type=int, help="WM focus cooldown"
    )
    parser.add_argument(
        "--early_stop_ep_length", type=int, help="Early stop episode length"
    )
    parser.add_argument("--eval_every", type=int, help="Eval interval")
    parser.add_argument("--eval_episodes", type=int, help="Eval episodes")
    parser.add_argument("--checkpoint_interval", type=int, help="Checkpoint interval")
    parser.add_argument("--num_collectors", type=int, help="Number of collectors")
    parser.add_argument("--replay_buffer_size", type=int, help="Replay buffer size")
    parser.add_argument("--min_buffer_episodes", type=int, help="Min buffer episodes")
    parser.add_argument(
        "--steps_per_weight_sync", type=int, help="Steps per weight sync"
    )
    parser.add_argument(
        "--replay_burn_in", type=int, help="Replay burn-in steps per sequence"
    )
    parser.add_argument("--replay_ratio", type=float, help="Replay ratio")
    parser.add_argument("--action_repeat", type=int, help="Action repeat")
    parser.add_argument("--recent_fraction", type=float, help="Recent fraction")
    parser.add_argument("--baseline_mode", type=bool, help="Baseline mode")

    return parser


def apply_cli_args(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides to Config."""
    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is not None and hasattr(cfg, k) and k not in ["config", "dry_run"]:
            setattr(cfg, k, v)
    return cfg


def run_training(cfg: Config, checkpoint_path: str | None = None):
    """Run training with the given configuration."""
    ensure_single_trainer()
    device = resolve_device(cfg.device)

    if cfg.dry_run:
        log_dir = Path(tempfile.mkdtemp(prefix="dreamer_dry_"))
        timestamp = datetime.now().strftime("%m-%d_%H%M%S")
        print(f"DRY RUN MODE - no MLflow, no checkpoints, temp dir: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%m-%d_%H%M%S")
        log_dir = Path("runs") / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training Configuration:")
    print(f"  Environment: {cfg.environment_name}")
    print(f"  Device: {device}")
    print(f"  d_hidden: {cfg.d_hidden}")
    print(f"  batch_size: {cfg.batch_size}")
    print(f"  actor_lr: {cfg.actor_lr}")
    print(f"  actor_entropy_coef: {cfg.actor_entropy_coef}")
    print(f"  Output: {log_dir}")

    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")

    # Dump config JSON for repro
    config_path = log_dir / "config.json"
    dump_config_json(cfg, str(config_path))
    print(f"  Config saved: {config_path}")

    # Setup MLflow
    mlflow_run_id = None
    if not cfg.dry_run:
        mlruns_dir = log_dir.parent / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlruns_dir.resolve()}")

        exp_name = cfg.experiment_name or f"DreamerV3-{cfg.environment_name}"
        mlflow.set_experiment(exp_name)

        run_name = f"{cfg.experiment_name or 'run'}_{timestamp}"
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
            if cfg.use_pixels and cfg.n_observations > 0
            else "vision"
            if cfg.use_pixels
            else "vector"
        )
        mlflow.set_tag("obs_mode", obs_mode)

        start_mlflow_ui(str(mlruns_dir), port=5000)

        # Log config as params
        for k, v in asdict(cfg).items():
            if isinstance(v, str) and len(v) > 250:
                v = v[:250] + "..."
            mlflow.log_param(k, str(v))

    try:
        mp_ctx = mp.get_context("spawn")
        data_queue = mp_ctx.Queue(maxsize=cfg.batch_size * 5)
        model_queue = mp_ctx.Queue(maxsize=1)
        stop_event = mp_ctx.Event()

        experience_loops = []
        for _ in range(max(1, cfg.num_collectors)):
            p = mp_ctx.Process(
                target=collect_experiences,
                args=(data_queue, model_queue, cfg, stop_event, str(log_dir)),
            )
            experience_loops.append(p)
        trainer_loop = mp_ctx.Process(
            target=train_world_model,
            args=(
                cfg,
                data_queue,
                model_queue,
                str(log_dir),
                checkpoint_path,
                mlflow_run_id,
                cfg.dry_run,
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
        if not cfg.dry_run:
            mlflow.end_run()


def main():
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Apply config based on --config flag
    if args.config == "cartpole":
        cfg = cartpole_config()
    elif args.config == "ratio_sweep_5e4":
        cfg = ratio_sweep_5e4_config()
    elif args.config == "paper_cartpole":
        cfg = paper_cartpole_config()
    elif args.config == "atari_pong":
        cfg = atari_pong_config()
    else:
        cfg = default_config()

    # Apply dry_run if passed
    if args.dry_run:
        cfg.dry_run = True

    # Apply CLI overrides
    cfg = apply_cli_args(cfg, args)

    run_training(cfg, checkpoint_path=args.checkpoint_path)


if __name__ == "__main__":
    main()
