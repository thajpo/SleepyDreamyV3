"""
Flat dataclass-based config for DreamerV3 training.

Design:
- Single flat dataclass with all training knobs
- Functions for base configs and experiments ("function pattern")
- Flat CLI arguments via argparse (e.g., --wm_lr 5e-4)
- JSON snapshots written to run_dir/config.json for reproducibility
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    """DreamerV3 configuration (flat structure)."""

    # ===== General settings =====
    device: str = "auto"  # auto, cuda, mps, cpu
    use_pixels: bool = False
    profile: bool = False
    compile_models: bool = False
    dry_run: bool = False
    experiment_name: Optional[str] = None

    # ===== Environment =====
    environment_name: str = "CartPole-v1"
    n_actions: int = 2
    n_observations: int = 4

    # ===== Model architecture =====
    d_hidden: int = 64
    num_latents: int = 32

    # Encoder CNN (pixels only)
    encoder_cnn_stride: int = 2
    encoder_cnn_kernel_size: int = 2
    encoder_cnn_padding: int = 0
    encoder_cnn_input_channels: int = 3
    encoder_cnn_num_layers: int = 4
    encoder_cnn_final_feature_size: int = 4
    encoder_cnn_target_size: tuple = (64, 64)

    # Encoder MLP
    encoder_mlp_hidden_dim_ratio: int = 8
    encoder_mlp_n_layers: int = 3

    # RNN
    rnn_n_blocks: int = 4

    # ===== Training: core settings =====
    max_train_steps: int = 10000
    batch_size: int = 16
    sequence_length: int = 64

    # ===== Training: learning rates (paper: 4e-5 with LaProp) =====
    wm_lr: float = 4e-5
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5

    # ===== Training: optimizer settings =====
    weight_decay: float = 0.0
    critic_ema_decay: float = 0.98
    critic_ema_regularizer: float = 1.0
    critic_replay_scale: float = 0.3

    # ===== Training: RL settings =====
    gamma: float = 0.997
    lam: float = 0.95
    num_dream_steps: int = 15
    actor_entropy_coef: float = 3e-4
    normalize_advantages: bool = True

    # ===== Training: loss coefficients =====
    beta_dyn: float = 1.0
    beta_rep: float = 0.1
    beta_pred: float = 1.0

    # ===== Training: reward bins =====
    b_start: int = -20
    b_end: int = 20

    # ===== Training: WM-AC ratio =====
    wm_ac_ratio: int = 1

    # ===== Training: LR schedule =====
    lr_cosine_decay: bool = False
    lr_cosine_min_factor: float = 0.1

    # ===== Training: WM-AC ratio schedule =====
    wm_ac_ratio_cosine: bool = False
    wm_ac_ratio_max: int = 8
    wm_ac_ratio_min: int = 2
    wm_ac_ratio_invert: bool = False

    # ===== Training: surprise-scaled AC LR =====
    surprise_scale_ac_lr: bool = False
    surprise_lr_scale_k: float = 10.0
    surprise_ema_beta: float = 0.99

    # ===== Training: WM focus mode =====
    surprise_wm_focus_threshold: float = 0.05
    surprise_wm_focus_ratio: int = 4
    surprise_wm_focus_duration: int = 20
    surprise_wm_focus_cooldown: int = 50

    # ===== Training: early stopping =====
    early_stop_ep_length: int = 0

    # ===== Training: evaluation =====
    eval_every: int = 1000
    eval_episodes: int = 5

    # ===== Training: checkpointing =====
    checkpoint_interval: int = 5000

    # ===== Training: data collection =====
    num_collectors: int = 1
    replay_buffer_size: int = 100000
    min_buffer_episodes: int = 64
    steps_per_weight_sync: int = 5

    # ===== Training: replay ratio gating =====
    replay_ratio: float = 1.0
    action_repeat: int = 1
    recent_fraction: float = 0.0

    # ===== Training: baseline mode =====
    baseline_mode: bool = True


def default_config() -> Config:
    """Default Dreamer config (paper values, slow/stable)."""
    return Config()


def cartpole_config() -> Config:
    """CartPole config for fast iteration."""
    cfg = default_config()
    cfg.environment_name = "CartPole-v1"
    cfg.n_actions = 2
    cfg.n_observations = 4
    cfg.d_hidden = 128
    cfg.max_train_steps = 10000
    cfg.batch_size = 8
    cfg.sequence_length = 16
    cfg.num_dream_steps = 10
    cfg.actor_entropy_coef = 1e-3
    cfg.wm_ac_ratio = 1
    cfg.b_start = -5
    cfg.b_end = 6
    cfg.replay_buffer_size = 10000
    cfg.replay_ratio = 16
    cfg.min_buffer_episodes = 32
    cfg.recent_fraction = 0.2
    cfg.surprise_scale_ac_lr = False
    cfg.surprise_wm_focus_threshold = 1e9
    cfg.surprise_wm_focus_ratio = 1
    cfg.surprise_wm_focus_duration = 0
    cfg.baseline_mode = False
    cfg.num_collectors = 8
    cfg.eval_every = 1000
    cfg.checkpoint_interval = 10000
    return cfg


def ratio_sweep_5e4_config() -> Config:
    """Config for WM:AC ratio sweep at LR=5e-4."""
    cfg = cartpole_config()
    cfg.experiment_name = "ratio_sweep_5e4"
    cfg.wm_lr = 5e-4
    cfg.actor_lr = 5e-4
    cfg.critic_lr = 5e-4
    cfg.wm_ac_ratio = 1
    return cfg


def paper_cartpole_config() -> Config:
    """Paper-accurate CartPole config (slower but stable)."""
    cfg = default_config()
    cfg.environment_name = "CartPole-v1"
    cfg.n_actions = 2
    cfg.n_observations = 4
    cfg.d_hidden = 128  # Keep 128 for capacity
    cfg.max_train_steps = 25000
    cfg.batch_size = 16  # Paper value
    cfg.sequence_length = 64  # Paper value
    cfg.num_dream_steps = 15  # Paper value
    cfg.actor_entropy_coef = 3e-4  # Paper value (not 1e-3)
    cfg.wm_ac_ratio = 1
    cfg.b_start = -5
    cfg.b_end = 6
    cfg.replay_buffer_size = 10000
    cfg.replay_ratio = 1.0  # Paper value (not 16)
    cfg.min_buffer_episodes = 64  # Paper value
    cfg.recent_fraction = 0.0  # Paper value
    cfg.surprise_scale_ac_lr = False
    cfg.surprise_wm_focus_threshold = 1e9
    cfg.surprise_wm_focus_ratio = 1
    cfg.surprise_wm_focus_duration = 0
    cfg.baseline_mode = False
    cfg.num_collectors = 8
    cfg.eval_every = 1000
    cfg.checkpoint_interval = 5000  # Checkpoint every 5k steps
    cfg.experiment_name = "paper_cartpole_baseline"
    return cfg


def atari_pong_config() -> Config:
    """Atari Pong config for pixel-based training (conservative memory settings)."""
    cfg = default_config()
    cfg.environment_name = "ALE/Pong-v5"
    cfg.n_actions = (
        6  # Pong has 6 actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
    )
    cfg.n_observations = 0  # Pixel observations, not vector
    cfg.use_pixels = True
    cfg.d_hidden = 256  # Your working value from Lunar Lander
    cfg.max_train_steps = 500000  # 500k steps for Atari
    cfg.batch_size = 8  # Your working value
    cfg.sequence_length = 32  # Compromise between 25 and 64
    cfg.num_dream_steps = 15
    cfg.actor_entropy_coef = 3e-4
    cfg.wm_ac_ratio = 1
    cfg.b_start = -20  # Wider range for Atari rewards
    cfg.b_end = 20
    cfg.replay_buffer_size = 100000  # Larger buffer for Atari
    cfg.replay_ratio = 1.0
    cfg.min_buffer_episodes = 64
    cfg.recent_fraction = 0.0
    cfg.surprise_scale_ac_lr = False
    cfg.surprise_wm_focus_threshold = 1e9
    cfg.surprise_wm_focus_ratio = 1
    cfg.surprise_wm_focus_duration = 0
    cfg.baseline_mode = False
    cfg.num_collectors = 4  # Reduced from 8 to prevent memory crash
    cfg.eval_every = 10000  # Eval less frequently for Atari
    cfg.checkpoint_interval = 50000  # Checkpoint every 50k steps
    cfg.experiment_name = "atari_pong_v2"
    # Pixel-specific settings
    cfg.encoder_cnn_target_size = (64, 64)  # Resize to 64x64 for model
    # Safer hyperparameters for pixel training (prevent NaN)
    cfg.wm_lr = 2e-5  # Lower WM LR for stability (was 4e-5)
    cfg.actor_lr = 3e-5  # Slightly lower actor LR
    cfg.critic_lr = 3e-5  # Slightly lower critic LR
    return cfg


def dump_config_json(cfg: Config, path: str) -> None:
    """Dump config to JSON for reproducibility."""
    import json

    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
