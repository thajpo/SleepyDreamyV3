# config.py
from pydantic import BaseModel
from typing import Tuple
import torch


def get_default_device():
    """Checks for available hardware accelerators."""
    if torch.cuda.is_available():
        # This works for both NVIDIA (CUDA) and AMD (ROCm)
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GeneralConfig(BaseModel):
    debug_memory: bool = False
    device: str = get_default_device()
    encoder_path: str = "encoder.pt"
    world_model_path: str = "world_model.pt"
    train_world_model: bool = True
    env_bootstrapping_samples: str = "bootstrap_trajectorires.h5"
    profile: bool = False
    compile_models: bool = False


class EnvironmentConfig(BaseModel):
    environment_name: str = "LunarLander-v3"
    n_actions: int = 4
    n_observations: int = 8


class CNNEncoderConfig(BaseModel):
    stride: int = 2
    activation: str = "sigmoid"
    target_size: Tuple[int, int] = (64, 64)
    kernel_size: int = 2
    padding: int = 0
    input_channels: int = 3  # RGB
    num_layers: int = 4  # number of convolutional layers
    final_feature_size: int = 4  # output is final_feature_size x final_feature_size


class MLPEncoderConfig(BaseModel):
    hidden_dim_ratio: int = 8
    n_layers: int = 3
    latent_categories: int = 16  # Number of categories per latent variable


class GRUConfig(BaseModel):
    n_blocks: int = 4


class EncoderConfig(BaseModel):
    cnn: CNNEncoderConfig = CNNEncoderConfig()
    mlp: MLPEncoderConfig = MLPEncoderConfig()


class ModelsConfig(BaseModel):
    d_hidden: int = 256
    encoder: EncoderConfig = EncoderConfig()
    rnn: GRUConfig = GRUConfig()


class TrainConfig(BaseModel):
    sequence_length: int = 25
    max_train_steps: int = 100000
    num_dream_steps: int = 15
    gamma: float = 0.99
    lam: float = 0.95
    wm_lr: float = 1e-4
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 8
    steps_per_weight_sync: int = 5
    beta_dyn: float = 0.99
    beta_rep: float = 0.99
    beta_pred: float = 0.99
    b_start: int = -20
    b_end: int = 21
    # Actor training stabilization
    actor_entropy_coef: float = 1e-3  # Entropy regularization coefficient
    normalize_advantages: bool = True  # Normalize advantages to stabilize training
    actor_warmup_steps: int = 1000  # Bootstrap: WM-only training, random actions in collector
    bootstrap_steps: int = 50000  # Default steps for bootstrap mode
    num_collectors: int = 1  # Number of parallel environment collectors
    # Replay buffer settings
    replay_buffer_size: int = 1000  # Max episodes stored in replay buffer
    min_buffer_episodes: int = 64   # Wait for this many episodes before training starts


class Config(BaseModel):
    general: GeneralConfig = GeneralConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    models: ModelsConfig = ModelsConfig()
    train: TrainConfig = TrainConfig()


# Default configuration instance
config = Config()
