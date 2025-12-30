import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import subprocess
import atexit
import os
from datetime import datetime

from .config import config
from .trainer import train_world_model
from .environment import collect_experiences

_tensorboard_process = None

def start_tensorboard(logdir="runs", port=6006):
    """Start TensorBoard in background."""
    global _tensorboard_process
    try:
        _tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port), "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"TensorBoard started: http://localhost:{port}")
        atexit.register(lambda: _tensorboard_process.terminate() if _tensorboard_process else None)
    except FileNotFoundError:
        print("TensorBoard not found. Install with: pip install tensorboard")


def _run_training(args, mode, checkpoint_path=None, reset_ac=False):
    """Unified training launcher for all modes."""
    if hasattr(args, 'debug_memory') and args.debug_memory:
        config.general.debug_memory = True

    # Set steps based on mode
    if mode == 'bootstrap':
        config.train.max_train_steps = args.steps
    elif hasattr(args, 'train_steps') and args.train_steps:
        config.train.max_train_steps = args.train_steps

    if mode == 'train' and hasattr(args, 'warmup_steps') and args.warmup_steps is not None:
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

    experience_loop = mp.Process(target=collect_experiences, args=(data_queue, model_queue))
    trainer_loop = mp.Process(
        target=train_world_model,
        args=(config, data_queue, model_queue, log_dir, checkpoint_path, mode, reset_ac)
    )

    experience_loop.start()
    trainer_loop.start()

    experience_loop.join()
    trainer_loop.join()

    print("Both processes have finished.")


def main():
    """Initializes networks and processes for training/inference."""
    # Required for CUDA with multiprocessing
    mp.set_start_method('spawn', force=True)

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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Train subparser (backward compatible unified training)
    train_parser = subparsers.add_parser('train', help='Full training with warmup')
    train_parser.add_argument(
        '--train_steps', type=int, default=config.train.max_train_steps,
        help=f'Number of training steps (default: {config.train.max_train_steps})'
    )
    train_parser.add_argument(
        '--warmup_steps', type=int, default=config.train.actor_warmup_steps,
        help=f'WM-only warmup steps before actor-critic (default: {config.train.actor_warmup_steps})'
    )
    train_parser.add_argument('--debug_memory', action='store_true', help='Enable memory profiling')

    # Bootstrap subparser (WM-only training)
    bootstrap_parser = subparsers.add_parser('bootstrap', help='WM-only training with random actions')
    bootstrap_parser.add_argument(
        '--steps', type=int, default=config.train.bootstrap_steps,
        help=f'Number of bootstrap training steps (default: {config.train.bootstrap_steps})'
    )
    bootstrap_parser.add_argument('--debug_memory', action='store_true', help='Enable memory profiling')

    # Dreamer subparser (full AC training from checkpoint)
    dreamer_parser = subparsers.add_parser('dreamer', help='Full AC training from checkpoint')
    dreamer_parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to checkpoint file'
    )
    # Require explicit intent - no auto-detection
    dreamer_ac_group = dreamer_parser.add_mutually_exclusive_group(required=True)
    dreamer_ac_group.add_argument(
        '--reset-ac', action='store_true',
        help='Reset actor/critic to random (keep WM weights only)'
    )
    dreamer_ac_group.add_argument(
        '--resume', action='store_true',
        help='Resume all weights from checkpoint (WM + actor/critic)'
    )
    dreamer_parser.add_argument(
        '--train_steps', type=int, default=config.train.max_train_steps,
        help=f'Number of training steps (default: {config.train.max_train_steps})'
    )
    dreamer_parser.add_argument('--debug_memory', action='store_true', help='Enable memory profiling')

    # Deploy subparser (inference)
    deploy_parser = subparsers.add_parser('deploy', help='Deploy trained model for inference')

    args = parser.parse_args()

    if args.mode == 'train':
        _run_training(args, mode='train')
    elif args.mode == 'bootstrap':
        _run_training(args, mode='bootstrap')
    elif args.mode == 'dreamer':
        reset_ac = getattr(args, 'reset_ac', False)
        _run_training(args, mode='dreamer', checkpoint_path=args.checkpoint, reset_ac=reset_ac)
    elif args.mode == 'deploy':
        print("Deploy mode not yet implemented")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
