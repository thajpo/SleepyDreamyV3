# config.py - Hydra-compatible configuration
from dataclasses import dataclass, field
from typing import List
import torch


def get_default_device() -> str:
    """Checks for available hardware accelerators."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GeneralConfig:
    device: str = "auto"  # auto, cuda, mps, cpu
    use_pixels: bool = False
    profile: bool = False
    compile_models: bool = False
    debug_memory: bool = False

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            return get_default_device()
        return self.device


@dataclass
class EnvironmentConfig:
    name: str = "CartPole-v1"
    n_actions: int = 2
    n_observations: int = 4


@dataclass
class CNNEncoderConfig:
    stride: int = 2
    kernel_size: int = 2
    padding: int = 0
    input_channels: int = 3
    num_layers: int = 4
    final_feature_size: int = 4
    target_size: List[int] = field(default_factory=lambda: [64, 64])


@dataclass
class MLPEncoderConfig:
    hidden_dim_ratio: int = 8
    n_layers: int = 3
    latent_categories: int = 16


@dataclass
class EncoderConfig:
    cnn: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    mlp: MLPEncoderConfig = field(default_factory=MLPEncoderConfig)


@dataclass
class RNNConfig:
    n_blocks: int = 4


@dataclass
class ModelsConfig:
    d_hidden: int = 64
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)


@dataclass
class TrainConfig:
    # Core settings
    max_steps: int = 30000
    batch_size: int = 32
    sequence_length: int = 25

    # Learning rates
    wm_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    weight_decay: float = 1e-6

    # RL settings
    gamma: float = 0.99
    lam: float = 0.95
    num_dream_steps: int = 15
    actor_entropy_coef: float = 1e-3
    normalize_advantages: bool = True

    # Loss coefficients
    beta_dyn: float = 0.99
    beta_rep: float = 0.99
    beta_pred: float = 0.99

    # Reward bins
    b_start: int = -5
    b_end: int = 6

    # Training phases
    bootstrap_steps: int = 5000
    actor_warmup_steps: int = 5000

    # Surprise-scaled AC learning rate
    surprise_scale_ac_lr: bool = True
    surprise_lr_scale_k: float = 10.0

    # WM focus mode (extra WM steps when surprise high)
    surprise_wm_focus_threshold: float = 0.05
    surprise_wm_focus_ratio: int = 4
    surprise_wm_focus_duration: int = 20

    # Checkpointing
    checkpoint_interval: int = 5000

    # Data collection
    num_collectors: int = 1
    replay_buffer_size: int = 1000
    min_buffer_episodes: int = 64
    steps_per_weight_sync: int = 5

    # WM surprise metrics
    surprise_ema_beta: float = 0.99


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# For backward compatibility with code expecting config.models.encoder.cnn.target_size as tuple
class ConfigAdapter:
    """Adapter to make Hydra config work with existing code."""

    def __init__(self, cfg):
        self._cfg = cfg

    def model_dump_json(self, indent=2):
        """Serialize config to JSON (Pydantic-compatible interface)."""
        import json
        from omegaconf import OmegaConf
        return json.dumps(OmegaConf.to_container(self._cfg), indent=indent)

    @property
    def general(self):
        return GeneralAdapter(self._cfg.general)

    @property
    def environment(self):
        return EnvironmentAdapter(self._cfg.environment)

    @property
    def models(self):
        return ModelsAdapter(self._cfg.models)

    @property
    def train(self):
        return TrainAdapter(self._cfg.train)


class GeneralAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def device(self):
        if self._cfg.device == "auto":
            return get_default_device()
        return self._cfg.device

    @property
    def use_pixels(self):
        return self._cfg.use_pixels

    @property
    def profile(self):
        return self._cfg.profile

    @property
    def compile_models(self):
        return self._cfg.compile_models

    @property
    def debug_memory(self):
        return self._cfg.debug_memory

    @property
    def dry_run(self):
        return self._cfg.dry_run


class EnvironmentAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def environment_name(self):
        return self._cfg.name

    @property
    def n_actions(self):
        return self._cfg.n_actions

    @property
    def n_observations(self):
        return self._cfg.n_observations


class CNNAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def stride(self):
        return self._cfg.stride

    @property
    def kernel_size(self):
        return self._cfg.kernel_size

    @property
    def padding(self):
        return self._cfg.padding

    @property
    def input_channels(self):
        return self._cfg.input_channels

    @property
    def num_layers(self):
        return self._cfg.num_layers

    @property
    def final_feature_size(self):
        return self._cfg.final_feature_size

    @property
    def target_size(self):
        # Return as tuple for backward compatibility
        return tuple(self._cfg.target_size)


class MLPAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def hidden_dim_ratio(self):
        return self._cfg.hidden_dim_ratio

    @property
    def n_layers(self):
        return self._cfg.n_layers

    @property
    def latent_categories(self):
        return self._cfg.latent_categories


class EncoderAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def cnn(self):
        return CNNAdapter(self._cfg.cnn)

    @property
    def mlp(self):
        return MLPAdapter(self._cfg.mlp)


class RNNAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def n_blocks(self):
        return self._cfg.n_blocks


class ModelsAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def d_hidden(self):
        return self._cfg.d_hidden

    @property
    def encoder(self):
        return EncoderAdapter(self._cfg.encoder)

    @property
    def rnn(self):
        return RNNAdapter(self._cfg.rnn)


class TrainAdapter:
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def max_train_steps(self):
        return self._cfg.max_steps

    @property
    def batch_size(self):
        return self._cfg.batch_size

    @property
    def sequence_length(self):
        return self._cfg.sequence_length

    @property
    def wm_lr(self):
        return self._cfg.wm_lr

    @property
    def actor_lr(self):
        return self._cfg.actor_lr

    @property
    def critic_lr(self):
        return self._cfg.critic_lr

    @property
    def weight_decay(self):
        return self._cfg.weight_decay

    @property
    def gamma(self):
        return self._cfg.gamma

    @property
    def lam(self):
        return self._cfg.lam

    @property
    def num_dream_steps(self):
        return self._cfg.num_dream_steps

    @property
    def actor_entropy_coef(self):
        return self._cfg.actor_entropy_coef

    @property
    def normalize_advantages(self):
        return self._cfg.normalize_advantages

    @property
    def beta_dyn(self):
        return self._cfg.beta_dyn

    @property
    def beta_rep(self):
        return self._cfg.beta_rep

    @property
    def beta_pred(self):
        return self._cfg.beta_pred

    @property
    def b_start(self):
        return self._cfg.b_start

    @property
    def b_end(self):
        return self._cfg.b_end

    @property
    def bootstrap_steps(self):
        return self._cfg.bootstrap_steps

    @property
    def actor_warmup_steps(self):
        return self._cfg.actor_warmup_steps

    @property
    def num_collectors(self):
        return self._cfg.num_collectors

    @property
    def replay_buffer_size(self):
        return self._cfg.replay_buffer_size

    @property
    def min_buffer_episodes(self):
        return self._cfg.min_buffer_episodes

    @property
    def steps_per_weight_sync(self):
        return self._cfg.steps_per_weight_sync

    @property
    def surprise_ema_beta(self):
        return self._cfg.surprise_ema_beta

    @property
    def surprise_scale_ac_lr(self):
        return self._cfg.surprise_scale_ac_lr

    @property
    def surprise_lr_scale_k(self):
        return self._cfg.surprise_lr_scale_k

    @property
    def surprise_wm_focus_threshold(self):
        return self._cfg.surprise_wm_focus_threshold

    @property
    def surprise_wm_focus_ratio(self):
        return self._cfg.surprise_wm_focus_ratio

    @property
    def surprise_wm_focus_duration(self):
        return self._cfg.surprise_wm_focus_duration

    @property
    def checkpoint_interval(self):
        return self._cfg.checkpoint_interval


def adapt_config(cfg) -> ConfigAdapter:
    """Wrap Hydra config with adapter for backward compatibility."""
    return ConfigAdapter(cfg)


# Default config instance for backward compatibility with old code
# This is used by modules that import `from .config import config`
config = Config()
