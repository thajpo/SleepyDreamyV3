# Set AMD ROCm env var before any torch imports (needed for spawned subprocesses)
import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import argparse
import multiprocessing as mp
import subprocess
import atexit
from datetime import datetime

from .config import config
from .trainer import train_world_model
from .environment import collect_experiences
from .utils import load_env_config


_tensorboard_process = None


def start_tensorboard(logdir="runs", port=6006):
    """Start TensorBoard in background."""
    global _tensorboard_process
    try:
        _tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port), "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"TensorBoard started: http://localhost:{port}")
        atexit.register(
            lambda: _tensorboard_process.terminate() if _tensorboard_process else None
        )
    except FileNotFoundError:
        print("TensorBoard not found. Install with: pip install tensorboard")


def _run_training(args, mode, checkpoint_path=None, reset_ac=False):
    """Unified training launcher for all modes."""
    if hasattr(args, "debug_memory") and args.debug_memory:
        config.general.debug_memory = True
    if hasattr(args, "profile"):
        config.general.profile = args.profile
    if hasattr(args, "compile"):
        config.general.compile_models = args.compile

    # CLI args override config (if provided)
    if mode == "bootstrap" and hasattr(args, "steps") and args.steps:
        config.train.max_train_steps = args.steps
    else:
        # Use bootstrap_steps from config for bootstrap mode
        if mode == "bootstrap":
            config.train.max_train_steps = config.train.bootstrap_steps

    if hasattr(args, "train_steps") and args.train_steps:
        config.train.max_train_steps = args.train_steps

    if hasattr(args, "warmup_steps") and args.warmup_steps is not None:
        config.train.actor_warmup_steps = args.warmup_steps

    # Generate unique run ID for this session
    run_id = datetime.now().strftime("%m-%d_%H%M")
    log_dir = f"runs/{run_id}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Mode: {mode} | Steps: {config.train.max_train_steps}")
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")

    start_tensorboard(logdir=log_dir)
    print("Starting experience collection and training processes...")

    # Create queues for inter-process communication
    data_queue = mp.Queue(maxsize=config.train.batch_size * 5)
    model_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

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
    stop_event.set()  # Signal collector to stop
    experience_loop.join(timeout=5.0)
    if experience_loop.is_alive():
        experience_loop.terminate()

    print("Both processes have finished.")


def main():
    """Initializes networks and processes for training/inference."""
    # Required for CUDA with multiprocessing
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="""DreamerV3 Training Interface

Two-phase training workflow:
  1. bootstrap  - Train world model with random actions (learns dynamics)
  2. dreamer    - Train actor-critic on learned world model (learns policy)

Or use 'train' for unified training with warmup period.

Examples:
  python -m src.main bootstrap --steps 50000
  python -m src.main dreamer --checkpoint runs/MM-DD_HHMM/checkpoints/wm_checkpoint_final.pt
  python -m src.main train --train_steps 100000
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Train subparser (backward compatible unified training)
    train_parser = subparsers.add_parser("train", help="Full training with warmup")
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to env config YAML (e.g., env_configs/cartpole.yaml)",
    )
    train_parser.add_argument(
        "--train_steps", type=int, help="Number of training steps (overrides config)"
    )
    train_parser.add_argument(
        "--warmup_steps",
        type=int,
        help="WM-only warmup steps before actor-critic (overrides config)",
    )
    train_parser.add_argument(
        "--debug_memory", action="store_true", help="Enable memory profiling"
    )
    train_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler for the full run",
    )
    train_parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for trainer models",
    )

    # Bootstrap subparser (WM-only training)
    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="WM-only training with random actions"
    )
    bootstrap_parser.add_argument(
        "--config",
        type=str,
        help="Path to env config YAML (e.g., env_configs/cartpole.yaml)",
    )
    bootstrap_parser.add_argument(
        "--steps",
        type=int,
        help="Number of bootstrap training steps (overrides config)",
    )
    bootstrap_parser.add_argument(
        "--debug_memory", action="store_true", help="Enable memory profiling"
    )
    bootstrap_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler for the full run",
    )
    bootstrap_parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for trainer models",
    )

    # Dreamer subparser (full AC training from checkpoint)
    dreamer_parser = subparsers.add_parser(
        "dreamer", help="Full AC training from checkpoint"
    )
    dreamer_parser.add_argument(
        "--config",
        type=str,
        help="Path to env config YAML (e.g., env_configs/cartpole.yaml)",
    )
    dreamer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    # Require explicit intent - no auto-detection
    dreamer_ac_group = dreamer_parser.add_mutually_exclusive_group(required=True)
    dreamer_ac_group.add_argument(
        "--reset-ac",
        action="store_true",
        help="Reset actor/critic to random (keep WM weights only)",
    )
    dreamer_ac_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume all weights from checkpoint (WM + actor/critic)",
    )
    dreamer_parser.add_argument(
        "--train_steps",
        type=int,
        default=config.train.max_train_steps,
        help=f"Number of training steps (default: {config.train.max_train_steps})",
    )
    dreamer_parser.add_argument(
        "--debug_memory", action="store_true", help="Enable memory profiling"
    )
    dreamer_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler for the full run",
    )
    dreamer_parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for trainer models",
    )

    # Deploy subparser (inference)
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy trained model for inference"
    )

    args = parser.parse_args()

    # Load env config if provided (before any mode-specific logic)
    if hasattr(args, "config") and args.config:
        load_env_config(args.config)

    if args.mode == "train":
        _run_training(args, mode="train")
    elif args.mode == "bootstrap":
        _run_training(args, mode="bootstrap")
    elif args.mode == "dreamer":
        reset_ac = getattr(args, "reset_ac", False)
        _run_training(
            args, mode="dreamer", checkpoint_path=args.checkpoint, reset_ac=reset_ac
        )
    elif args.mode == "deploy":
        print("Deploy mode not yet implemented")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
