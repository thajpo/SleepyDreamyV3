"""
Flat dataclass-based config for DreamerV3 training.

Design:
- Hydra YAML is the authored configuration source for training.
- This flat dataclass is the typed runtime projection and snapshot schema.
- JSON snapshots are written to each run directory for reproducibility.
"""

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


class ConfigValidationError(ValueError):
    """Raised before side effects when a training configuration is impossible."""


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
    log_profile: str = "lean"  # lean, full
    research_gradient_diagnostics: bool = False
    seed: int = 0

    # ===== Environment =====
    environment_name: str = "CartPole-v1"
    n_actions: int = 2
    n_observations: int = 4
    # Atari compatibility controls (used when environment_name starts with "ALE/")
    atari_compat_mode: bool = False
    atari_noop_max: int = 30
    atari_frame_skip: int = 4
    atari_terminal_on_life_loss: bool = False
    atari_sticky_action_prob: float = 0.25
    atari_full_action_space: bool = False
    atari_fire_reset: bool = False
    atari_screen_size: int = 84

    # ===== Model architecture =====
    d_hidden: int = 64
    num_latents: int = 32
    # Legacy preserves historical checkpoint construction. Authored Hydra
    # configs select the grouped, normalized reference recurrent core.
    rssm_core: str = "legacy"  # legacy, reference
    # Zero preserves historical linear-head checkpoint construction. Authored
    # Hydra configs use the reference-style one-hidden-layer continuation MLP.
    continue_head_layers: int = 0

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
    # Legacy preserves historical split-rate, detached replay-value updates.
    # Authored Hydra runs use the coherent reference optimizer contract.
    optimizer_contract: str = "legacy"  # legacy, reference
    laprop_bias_correction: bool = True
    optimizer_warmup_steps: int = 0
    weight_decay: float = 0.0
    critic_ema_decay: float = 0.98
    critic_ema_regularizer: float = 1.0
    # Distribution preserves historical full-logit matching. Authored Hydra
    # runs re-encode the slow critic's decoded scalar like reference DreamerV3.
    critic_ema_target: str = "distribution"  # distribution, mean_twohot
    # True preserves runs authored before the reference slow-target audit. Hydra
    # explicitly disables this for new training runs.
    critic_slow_target: bool = True
    critic_replay_scale: float = 0.3
    # Optional full-episode replay return-to-go supervision.
    critic_real_return_scale: float = 0.0
    q_critic_scale: float = 0.0
    q_actor_temperature: float = 0.25

    # ===== Training: RL settings =====
    gamma: float = 0.997
    lam: float = 0.95
    horizon: int = 333
    contdisc: bool = True
    num_dream_steps: int = 15
    actor_entropy_coef: float = 3e-4
    normalize_advantages: bool = False
    actor_loss_mode: str = "reinforce"  # reinforce, enumerate, qcritic, mpc_teacher
    actor_enum_horizon: int = 3
    actor_enum_temperature: float = 0.25
    actor_enum_loss_scale: float = 1.0
    actor_enum_objective: str = "value"  # value, survival
    mpc_teacher_horizon: int = 6
    mpc_teacher_temperature: float = 0.1
    mpc_teacher_loss_scale: float = 1.0
    mpc_teacher_objective: str = "value"  # value, survival
    mpc_teacher_target: str = "hard"  # hard, soft
    mpc_teacher_margin_min: float = 0.0
    mpc_teacher_normalize_values: bool = True
    terminal_reward_penalty: float = 0.0
    balance_continuation: bool = False
    continuation_balance_rate: float = 0.01
    # False preserves historical unit-prefix imagination-loss weighting.
    # Authored Hydra runs include the observed start continuation like DreamerV3.
    weight_imagination_starts: bool = False

    # ===== Training: loss coefficients =====
    beta_dyn: float = 1.0
    beta_rep: float = 0.1
    beta_pred: float = 1.0
    # Historical runs used half the mean squared vector-observation error.
    # Authored Hydra runs sum squared event errors like reference DreamerV3.
    state_loss_mode: str = "legacy_half_mean"  # legacy_half_mean, reference_sum
    free_bits_straight_through: bool = False
    prior_state_pred_scale: float = 0.0

    # ===== Training: reward bins =====
    b_start: int = -20
    b_end: int = 20
    num_bins: int = 255

    # ===== Training: WM-AC ratio =====
    wm_ac_ratio: int = 1
    actor_warmup_steps: int = 0

    # ===== Training: LR schedule =====
    lr_cosine_decay: bool = False
    lr_cosine_min_factor: float = 0.1

    # ===== Training: WM-AC ratio schedule =====
    wm_ac_ratio_cosine: bool = False
    wm_ac_ratio_max: int = 8
    wm_ac_ratio_min: int = 2
    wm_ac_ratio_invert: bool = False

    # ===== Training: early stopping =====
    early_stop_ep_length: int = 0

    # ===== Training: evaluation =====
    eval_every: int = 1000
    eval_episodes: int = 5
    eval_metric: str = "episode_reward"  # episode_reward, episode_length, win_rate

    # ===== Training: checkpointing =====
    checkpoint_interval: int = 5000

    # ===== Training: data collection =====
    num_collectors: int = 1
    replay_buffer_size: int = 500
    min_buffer_episodes: int = 64
    steps_per_weight_sync: int = 5
    replay_burn_in: int = 8
    # Episode preserves historical replay snapshots. Authored Hydra runs use
    # reference-style per-collector streams that can cross reset boundaries.
    replay_sequence_mode: str = "episode"  # episode, stream

    # ===== Training: replay ratio gating =====
    replay_ratio: float = 1.0
    action_repeat: int = 1
    recent_fraction: float = 0.0


def default_config() -> Config:
    """Legacy inspector fallback when a checkpoint has no config snapshot."""
    return Config()


def config_from_snapshot(data: dict) -> Config:
    """Construct a config, treating a missing LaProp mode as uncorrected."""
    normalized = dict(data)
    normalized.setdefault("laprop_bias_correction", False)
    return Config(**normalized)


def load_checkpoint_config(
    checkpoint_path: str | Path,
    checkpoint: dict | None = None,
) -> Config | None:
    """Load the authored config carried by or stored beside a checkpoint."""
    if checkpoint is not None:
        snapshot = checkpoint.get("config_snapshot")
        if isinstance(snapshot, dict):
            return config_from_snapshot(snapshot)

    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent.parent / "config.json"
    if config_path.exists():
        return config_from_snapshot(json.loads(config_path.read_text()))

    if checkpoint is None and checkpoint_path.exists():
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint is None:
            return None
        snapshot = checkpoint.get("config_snapshot")
        if isinstance(snapshot, dict):
            return config_from_snapshot(snapshot)
    return None


def atari100k_pong_config() -> Config:
    """Legacy inspector fallback for old Pong checkpoints without config.json."""
    cfg = default_config()
    cfg.environment_name = "ALE/Pong-v5"
    cfg.n_actions = 6
    cfg.n_observations = 0  # Pixel observations, not vector
    cfg.use_pixels = True
    cfg.atari_compat_mode = True
    cfg.atari_noop_max = 30
    cfg.atari_frame_skip = 4
    cfg.atari_terminal_on_life_loss = False
    cfg.atari_sticky_action_prob = 0.0
    cfg.atari_full_action_space = False
    cfg.atari_fire_reset = False
    cfg.atari_screen_size = 64
    cfg.action_repeat = 1
    cfg.d_hidden = 256
    cfg.max_train_steps = 110000
    cfg.batch_size = 16
    cfg.sequence_length = 64
    cfg.num_dream_steps = 15
    cfg.actor_entropy_coef = 3e-4
    cfg.wm_ac_ratio = 1
    cfg.b_start = -20
    cfg.b_end = 20
    cfg.replay_buffer_size = 1000
    cfg.replay_ratio = 256.0
    cfg.min_buffer_episodes = 16
    cfg.num_collectors = 1
    cfg.eval_every = 10000
    cfg.checkpoint_interval = 10000
    cfg.experiment_name = "atari100k_pong"
    cfg.encoder_cnn_target_size = (64, 64)
    cfg.wm_lr = 4e-5
    cfg.actor_lr = 4e-5
    cfg.critic_lr = 4e-5
    return cfg


def atari_pong_config() -> Config:
    """Backwards-compatible alias for Atari100k Pong config."""
    return atari100k_pong_config()


def validate_config(cfg: Config) -> None:
    """Reject invalid training configurations before creating run side effects."""
    errors: list[str] = []

    positive_ints = {
        "max_train_steps": cfg.max_train_steps,
        "batch_size": cfg.batch_size,
        "sequence_length": cfg.sequence_length,
        "d_hidden": cfg.d_hidden,
        "num_latents": cfg.num_latents,
        "rnn_n_blocks": cfg.rnn_n_blocks,
        "n_actions": cfg.n_actions,
        "num_dream_steps": cfg.num_dream_steps,
        "horizon": cfg.horizon,
        "checkpoint_interval": cfg.checkpoint_interval,
        "num_collectors": cfg.num_collectors,
        "replay_buffer_size": cfg.replay_buffer_size,
        "min_buffer_episodes": cfg.min_buffer_episodes,
        "steps_per_weight_sync": cfg.steps_per_weight_sync,
        "action_repeat": cfg.action_repeat,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            errors.append(f"{name} must be > 0 (got {value!r})")

    if cfg.d_hidden < 16 or cfg.d_hidden % 16 != 0:
        errors.append("d_hidden must be at least 16 and divisible by 16")
    if cfg.rssm_core not in {"legacy", "reference"}:
        errors.append("rssm_core must be 'legacy' or 'reference'")
    if cfg.continue_head_layers not in {0, 1}:
        errors.append("continue_head_layers must be 0 or 1")
    if cfg.optimizer_contract not in {"legacy", "reference"}:
        errors.append("optimizer_contract must be 'legacy' or 'reference'")
    if cfg.critic_ema_target not in {"distribution", "mean_twohot"}:
        errors.append("critic_ema_target must be 'distribution' or 'mean_twohot'")
    if cfg.state_loss_mode not in {"legacy_half_mean", "reference_sum"}:
        errors.append(
            "state_loss_mode must be 'legacy_half_mean' or 'reference_sum'"
        )
    if cfg.optimizer_warmup_steps < 0:
        errors.append("optimizer_warmup_steps must be >= 0")
    if cfg.actor_warmup_steps < 0:
        errors.append("actor_warmup_steps must be >= 0")
    if cfg.optimizer_contract == "reference":
        rates = (cfg.wm_lr, cfg.actor_lr, cfg.critic_lr)
        if not math.isclose(rates[0], rates[1]) or not math.isclose(
            rates[0], rates[2]
        ):
            errors.append(
                "reference optimizer_contract requires equal wm/actor/critic rates"
            )
    if not 0 <= cfg.replay_burn_in < cfg.sequence_length:
        errors.append("replay_burn_in must satisfy 0 <= burn-in < sequence_length")
    if cfg.min_buffer_episodes > cfg.replay_buffer_size:
        errors.append("min_buffer_episodes cannot exceed replay_buffer_size")
    if cfg.replay_sequence_mode not in {"episode", "stream"}:
        errors.append("replay_sequence_mode must be 'episode' or 'stream'")
    if cfg.replay_ratio <= 0:
        errors.append("replay_ratio must be > 0")
    if not 0.0 <= cfg.recent_fraction <= 1.0:
        errors.append("recent_fraction must be between 0 and 1")
    if not 0.0 < cfg.gamma <= 1.0:
        errors.append("gamma must be in (0, 1]")
    if cfg.contdisc and cfg.horizon > 0:
        continuation_discount = 1.0 - 1.0 / float(cfg.horizon)
        if not math.isclose(cfg.gamma, continuation_discount, abs_tol=1e-5):
            errors.append(
                "contdisc requires gamma to match 1 - 1 / horizon "
                f"(got gamma={cfg.gamma!r}, horizon={cfg.horizon!r})"
            )
    if not 0.0 <= cfg.lam <= 1.0:
        errors.append("lam must be between 0 and 1")
    if not 0.0 < cfg.continuation_balance_rate <= 1.0:
        errors.append("continuation_balance_rate must be in (0, 1]")
    if cfg.b_end <= cfg.b_start:
        errors.append("b_end must be greater than b_start")
    if cfg.num_bins < 2:
        errors.append("num_bins must be at least 2")
    if cfg.eval_every < 0 or cfg.eval_episodes < 0:
        errors.append("eval_every and eval_episodes cannot be negative")
    if cfg.eval_metric not in {"episode_length", "episode_reward", "win_rate"}:
        errors.append(f"unsupported eval_metric: {cfg.eval_metric!r}")
    if cfg.actor_loss_mode not in {
        "reinforce",
        "enumerate",
        "qcritic",
        "mpc_teacher",
    }:
        errors.append(f"unsupported actor_loss_mode: {cfg.actor_loss_mode!r}")
    if cfg.actor_enum_objective not in {"value", "survival"}:
        errors.append(f"unsupported actor_enum_objective: {cfg.actor_enum_objective!r}")
    if cfg.mpc_teacher_objective not in {"value", "survival"}:
        errors.append(f"unsupported mpc_teacher_objective: {cfg.mpc_teacher_objective!r}")
    if cfg.mpc_teacher_target not in {"hard", "soft"}:
        errors.append(f"unsupported mpc_teacher_target: {cfg.mpc_teacher_target!r}")
    if cfg.device not in {"auto", "cpu", "cuda", "mps"}:
        errors.append(f"unsupported device: {cfg.device!r}")
    if cfg.log_profile not in {"lean", "full"}:
        errors.append(f"unsupported log_profile: {cfg.log_profile!r}")
    if cfg.use_pixels:
        target = cfg.encoder_cnn_target_size
        if (
            not isinstance(target, (list, tuple))
            or len(target) != 2
            or any(int(dimension) <= 0 for dimension in target)
        ):
            errors.append("pixel runs require a positive two-dimensional target_size")
        if cfg.encoder_cnn_input_channels <= 0:
            errors.append("pixel runs require encoder_cnn_input_channels > 0")
    elif cfg.n_observations <= 0:
        errors.append("state-only runs require n_observations > 0")

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ConfigValidationError(f"Invalid Dreamer configuration:\n{details}")


def dump_config_json(cfg: Config, path: str) -> None:
    """Dump config to JSON for reproducibility."""
    import json

    def to_jsonable(value):
        if hasattr(value, "items"):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        return value

    with open(path, "w") as f:
        json.dump(to_jsonable(asdict(cfg)), f, indent=2)
