"""Trainer package for DreamerV3 world model training."""

from .math_utils import symlog, symexp, twohot_encode, resize_pixels_to_target, unimix_logits
from .model_init import initialize_actor, initialize_critic, initialize_world_model
from .profiling import ProfilerManager, TimingAccumulator
from .losses import compute_wm_loss, compute_actor_critic_losses
from .dreaming import dream_sequence, calculate_lambda_returns
from .checkpoints import save_checkpoint, save_wm_only_checkpoint, load_checkpoint
from .visualization import log_metrics
from .core import WorldModelTrainer, train_world_model
from .optimizers import LaProp, adaptive_gradient_clipping

__all__ = [
    # core
    "WorldModelTrainer",
    "train_world_model",
    # math_utils
    "symlog",
    "symexp",
    "twohot_encode",
    "resize_pixels_to_target",
    "unimix_logits",
    # model_init
    "initialize_actor",
    "initialize_critic",
    "initialize_world_model",
    # profiling
    "ProfilerManager",
    "TimingAccumulator",
    # losses
    "compute_wm_loss",
    "compute_actor_critic_losses",
    # dreaming
    "dream_sequence",
    "calculate_lambda_returns",
    # checkpoints
    "save_checkpoint",
    "save_wm_only_checkpoint",
    "load_checkpoint",
    # visualization
    "log_metrics",
    # optimizers
    "LaProp",
    "adaptive_gradient_clipping",
]
