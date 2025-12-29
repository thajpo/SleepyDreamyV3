import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import subprocess
import atexit

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


def main():
    """Initializes networks and processes for training/inference."""
    # Required for CUDA with multiprocessing
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Training and inference interface for DreamerV3."
    )
    parser.add_argument(
        "mode",
        help="Specify 'train' to train a world model, and 'deploy' for inference on a pretrained checkpoint",
    )
    parser.add_argument(
        "--train_steps", type=int, default=config.train.max_train_steps,
        help=f"Number of training steps (default: {config.train.max_train_steps})"
    )
    parser.add_argument(
        "--debug_memory", action="store_true", help="Enable memory profiling prints"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=config.train.actor_warmup_steps,
        help=f"WM-only warmup steps before actor-critic training (default: {config.train.actor_warmup_steps})"
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.debug_memory:
            config.general.debug_memory = True
        if args.train_steps:
            config.train.max_train_steps = args.train_steps
        if args.warmup_steps is not None:
            config.train.actor_warmup_steps = args.warmup_steps

        print(f"Training: {config.train.max_train_steps} steps, warmup: {config.train.actor_warmup_steps} steps")
        start_tensorboard()
        print("Starting experience collection and training processes...")

        # Create a queue to pass data from collector to trainer
        # maxsize prevents the collector from running too far ahead and using up all the memory.
        data_queue = mp.Queue(maxsize=config.train.batch_size * 5)
        model_queue = mp.Queue(maxsize=1)
        experience_loop = mp.Process(target=collect_experiences, args=(data_queue,model_queue))
        trainer_loop = mp.Process(target=train_world_model, args=(config, data_queue, model_queue))

        experience_loop.start()
        trainer_loop.start()

        experience_loop.join()
        trainer_loop.join()

        print("Both processes have finished.")
    elif args.mode == "deploy":
        pass  # TODO


if __name__ == "__main__":
    main()
