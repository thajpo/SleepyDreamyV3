#!/usr/bin/env python3
"""
sweep_profile.py - GPU Memory Profiler for Concurrent Hyperparameter Sweeps

Profiles GPU memory usage during training to determine how many concurrent
sweeps can safely run on your hardware.

Usage:
    # Profile a specific config
    uv run python scripts/sweep_profile.py --config env_configs/cartpole_state_only.yaml

    # Profile multiple state-only configs for comparison
    uv run python scripts/sweep_profile.py --sweep

    # Custom safety margin (default 10%)
    uv run python scripts/sweep_profile.py --config env_configs/cartpole.yaml --safety_margin 0.15

The script profiles:
- Model parameter memory
- Optimizer state memory (Adam uses 2x model params)
- Activation memory during forward pass
- Gradient memory during backward pass
- Peak memory during a training step
"""

import argparse
import gc
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.config import config
from src.trainer_utils import (
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    symlog,
    twohot_encode,
)


def load_env_config(config_path: str):
    """Load YAML config and apply overrides to global config (same as main.py)."""
    with open(config_path, "r") as f:
        overrides = yaml.safe_load(f)

    if "general" in overrides:
        for key, value in overrides["general"].items():
            if hasattr(config.general, key):
                setattr(config.general, key, value)

    if "environment" in overrides:
        for key, value in overrides["environment"].items():
            if hasattr(config.environment, key):
                setattr(config.environment, key, value)

    if "models" in overrides:
        for key, value in overrides["models"].items():
            if hasattr(config.models, key):
                setattr(config.models, key, value)

    if "train" in overrides:
        for key, value in overrides["train"].items():
            if hasattr(config.train, key):
                setattr(config.train, key, value)

    return config


def bytes_to_mb(b: float) -> float:
    return b / (1024 * 1024)


def bytes_to_gb(b: float) -> float:
    return b / (1024 * 1024 * 1024)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_info() -> dict:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {"available": False, "device_name": "CPU", "total_memory": 0}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "available": True,
        "device_name": props.name,
        "total_memory": props.total_memory,
        "multi_processor_count": props.multi_processor_count,
    }


def reset_memory_stats():
    """Reset CUDA memory statistics and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_memory_stats() -> dict:
    """Get current memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak": 0}

    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak": torch.cuda.max_memory_allocated(),
    }


def create_dummy_batch(cfg, device: torch.device) -> dict:
    """Create a dummy batch for profiling."""
    batch_size = cfg.train.batch_size
    seq_len = cfg.train.sequence_length
    n_obs = cfg.environment.n_observations
    n_actions = cfg.environment.n_actions

    batch = {
        "state": torch.randn(batch_size, seq_len, n_obs, device=device),
        "action": F.one_hot(
            torch.randint(0, n_actions, (batch_size, seq_len), device=device),
            num_classes=n_actions,
        ).float(),
        "reward": torch.randn(batch_size, seq_len, device=device),
        "done": torch.zeros(batch_size, seq_len, device=device),
    }

    if cfg.general.use_pixels:
        target_size = cfg.models.encoder.cnn.target_size
        batch["pixels"] = torch.randint(
            0, 256,
            (batch_size, seq_len, 3, target_size[0], target_size[1]),
            device=device,
            dtype=torch.float32,
        )

    return batch


def profile_training_step(cfg, warmup_steps: int = 3) -> dict:
    """
    Profile a single training step and measure peak memory usage.

    Returns dict with memory stats and parameter counts.
    """
    device = torch.device(cfg.general.device)

    # Reset memory tracking
    reset_memory_stats()

    # Initialize models
    encoder, world_model = initialize_world_model(
        device, cfg, batch_size=cfg.train.batch_size
    )
    actor = initialize_actor(device, cfg)
    critic = initialize_critic(device, cfg)

    # Count parameters
    encoder_params = count_parameters(encoder)
    world_model_params = count_parameters(world_model)
    actor_params = count_parameters(actor)
    critic_params = count_parameters(critic)
    total_params = encoder_params + world_model_params + actor_params + critic_params

    # Initialize optimizers (Adam stores 2 states per param: momentum & variance)
    wm_params = list(encoder.parameters()) + list(world_model.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=cfg.train.wm_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.train.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.train.critic_lr)

    # Memory after model + optimizer initialization
    model_init_memory = get_memory_stats()

    # Create reward bins for loss computation
    b_start = cfg.train.b_start
    b_end = cfg.train.b_end
    B = torch.arange(b_start, b_end, device=device).float()
    B = torch.sign(B) * (torch.exp(torch.abs(B)) - 1)  # symexp

    # Warmup passes (for torch.compile and CUDA kernel caching)
    batch = create_dummy_batch(cfg, device)

    for _ in range(warmup_steps):
        # Simulate a forward pass
        _run_forward_pass(encoder, world_model, actor, critic, batch, cfg, B, device)
        reset_memory_stats()

    # Profile actual training step
    reset_memory_stats()

    # Forward pass
    loss, actor_loss, critic_loss = _run_forward_pass(
        encoder, world_model, actor, critic, batch, cfg, B, device
    )
    forward_memory = get_memory_stats()

    # Backward pass - world model
    loss.backward()
    backward_wm_memory = get_memory_stats()

    # Backward pass - actor/critic
    if actor_loss is not None:
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

    backward_full_memory = get_memory_stats()

    # Optimizer step
    wm_optimizer.step()
    actor_optimizer.step()
    critic_optimizer.step()
    wm_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    optimizer_step_memory = get_memory_stats()

    # Final peak
    peak_memory = get_memory_stats()["peak"]

    return {
        "config": {
            "batch_size": cfg.train.batch_size,
            "d_hidden": cfg.models.d_hidden,
            "seq_length": cfg.train.sequence_length,
            "n_gru_blocks": cfg.models.rnn.n_blocks,
            "use_pixels": cfg.general.use_pixels,
            "n_dream_steps": cfg.train.num_dream_steps,
        },
        "params": {
            "encoder": encoder_params,
            "world_model": world_model_params,
            "actor": actor_params,
            "critic": critic_params,
            "total": total_params,
        },
        "memory": {
            "model_init": model_init_memory["allocated"],
            "forward": forward_memory["peak"],
            "backward_wm": backward_wm_memory["peak"],
            "backward_full": backward_full_memory["peak"],
            "optimizer_step": optimizer_step_memory["peak"],
            "peak": peak_memory,
        },
    }


def _run_forward_pass(encoder, world_model, actor, critic, batch, cfg, B, device):
    """Run a forward pass simulating training."""
    batch_size = cfg.train.batch_size
    seq_len = cfg.train.sequence_length
    n_actions = cfg.environment.n_actions
    n_dream_steps = cfg.train.num_dream_steps

    # Reset world model hidden state
    world_model.h_prev = torch.zeros_like(world_model.h_prev)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    all_h_z = []

    # Process sequence
    for t in range(seq_len):
        if cfg.general.use_pixels:
            obs = {
                "pixels": batch["pixels"][:, t],
                "state": symlog(batch["state"][:, t]),
            }
        else:
            obs = symlog(batch["state"][:, t])

        action = batch["action"][:, t]
        reward = batch["reward"][:, t]

        # Encode observation
        posterior_logits = encoder(obs)
        posterior_dist = Categorical(logits=posterior_logits)

        # World model step
        obs_rec, reward_logits, _, h_z, prior_logits = world_model(
            posterior_dist, action
        )

        all_h_z.append(h_z)

        # Compute losses
        if cfg.general.use_pixels:
            pixel_target = obs["pixels"]
            pixel_loss = F.binary_cross_entropy_with_logits(
                obs_rec["pixels"], pixel_target, reduction="mean"
            )
            state_loss = F.mse_loss(obs_rec["state"], obs["state"])
            pred_loss = pixel_loss + state_loss
        else:
            state_loss = F.mse_loss(obs_rec["state"], obs)
            pred_loss = state_loss

        reward_target = twohot_encode(reward, B)
        reward_loss = F.cross_entropy(reward_logits, reward_target)

        # KL losses
        prior_dist = Categorical(logits=prior_logits)
        kl_dyn = torch.distributions.kl_divergence(
            Categorical(probs=posterior_dist.probs.detach()), prior_dist
        ).mean()
        kl_rep = torch.distributions.kl_divergence(
            posterior_dist, Categorical(probs=prior_dist.probs.detach())
        ).mean()

        step_loss = pred_loss + reward_loss + 0.5 * kl_dyn + 0.1 * kl_rep
        total_loss = total_loss + step_loss

    # Dream rollout for actor/critic (simplified)
    h_z_dream = all_h_z[-1].detach()
    h_dream = world_model.h_prev.detach()

    dream_values = []
    dream_log_probs = []

    for _ in range(n_dream_steps):
        # Actor predicts action
        action_logits = actor(h_z_dream)
        action_dist = Categorical(logits=action_logits)
        action_sample = action_dist.sample()
        action_onehot = F.one_hot(action_sample, num_classes=n_actions).float()

        # Get value estimate
        value_logits = critic(h_z_dream)
        dream_values.append(value_logits)
        dream_log_probs.append(action_dist.log_prob(action_sample))

        # Step world model in imagination
        z_flat = h_z_dream[:, world_model.d_hidden * cfg.models.rnn.n_blocks:]
        z_embed = world_model.z_embedding(z_flat)
        h_dream, prior_logits = world_model.step_dynamics(z_embed, action_onehot, h_dream)

        # Sample from prior for next state
        prior_dist = Categorical(logits=prior_logits)
        z_idx = prior_dist.sample()
        z_onehot = F.one_hot(z_idx, num_classes=cfg.models.d_hidden // 16).float()
        z_flat_new = z_onehot.view(batch_size, -1)
        h_z_dream = torch.cat([h_dream, z_flat_new], dim=-1)

    # Simple actor/critic loss (not full implementation, just for memory profiling)
    if dream_values:
        # dream_values: list of (batch, n_bins) tensors
        # dream_log_probs: list of (batch,) tensors
        values_tensor = torch.stack(dream_values)  # (n_dream_steps, batch, n_bins)
        log_probs_tensor = torch.stack(dream_log_probs)  # (n_dream_steps, batch)

        # Simple loss - just use mean value for profiling purposes
        value_mean = values_tensor.mean(dim=-1)  # (n_dream_steps, batch)
        actor_loss = -(log_probs_tensor * value_mean.detach()).mean()
        critic_loss = F.mse_loss(value_mean, torch.zeros_like(value_mean))
    else:
        actor_loss = None
        critic_loss = None

    return total_loss / seq_len, actor_loss, critic_loss


def calculate_concurrent_sweeps(
    peak_memory: int,
    total_gpu_memory: int,
    safety_margin: float = 0.10,
) -> dict:
    """
    Calculate how many concurrent sweeps can fit in GPU memory.

    Args:
        peak_memory: Peak memory used by one training run (bytes)
        total_gpu_memory: Total GPU memory available (bytes)
        safety_margin: Fraction of memory to reserve (default 10%)

    Returns:
        Dict with concurrent sweep calculations
    """
    usable_memory = total_gpu_memory * (1 - safety_margin)
    max_concurrent = int(usable_memory / peak_memory)
    utilization = (max_concurrent * peak_memory) / total_gpu_memory

    return {
        "max_concurrent": max_concurrent,
        "memory_per_sweep_mb": bytes_to_mb(peak_memory),
        "total_gpu_memory_gb": bytes_to_gb(total_gpu_memory),
        "usable_memory_gb": bytes_to_gb(usable_memory),
        "safety_margin": safety_margin,
        "estimated_utilization": utilization,
    }


def print_profile_report(result: dict, gpu_info: dict, concurrent: dict):
    """Print a formatted profiling report."""
    print("\n" + "=" * 70)
    print("  SWEEP PROFILER REPORT")
    print("=" * 70)

    # GPU Info
    print(f"\n{'GPU Information':─^50}")
    print(f"  Device:           {gpu_info['device_name']}")
    print(f"  Total Memory:     {bytes_to_gb(gpu_info['total_memory']):.2f} GB")

    # Config
    cfg = result["config"]
    print(f"\n{'Configuration':─^50}")
    print(f"  Batch Size:       {cfg['batch_size']}")
    print(f"  Hidden Dim:       {cfg['d_hidden']}")
    print(f"  Sequence Length:  {cfg['seq_length']}")
    print(f"  GRU Blocks:       {cfg['n_gru_blocks']}")
    print(f"  Dream Steps:      {cfg['n_dream_steps']}")
    print(f"  Use Pixels:       {cfg['use_pixels']}")

    # Parameters
    params = result["params"]
    print(f"\n{'Model Parameters':─^50}")
    print(f"  Encoder:          {params['encoder']:,}")
    print(f"  World Model:      {params['world_model']:,}")
    print(f"  Actor:            {params['actor']:,}")
    print(f"  Critic:           {params['critic']:,}")
    print(f"  {'─' * 30}")
    print(f"  Total:            {params['total']:,}")

    # Memory breakdown
    mem = result["memory"]
    print(f"\n{'Memory Usage':─^50}")
    print(f"  Model + Optimizer:  {bytes_to_mb(mem['model_init']):>8.1f} MB")
    print(f"  Forward Pass:       {bytes_to_mb(mem['forward']):>8.1f} MB")
    print(f"  WM Backward:        {bytes_to_mb(mem['backward_wm']):>8.1f} MB")
    print(f"  Full Backward:      {bytes_to_mb(mem['backward_full']):>8.1f} MB")
    print(f"  After Optimizer:    {bytes_to_mb(mem['optimizer_step']):>8.1f} MB")
    print(f"  {'─' * 30}")
    print(f"  Peak Memory:        {bytes_to_mb(mem['peak']):>8.1f} MB")

    # Concurrent sweep calculation
    print(f"\n{'Concurrent Sweep Capacity':─^50}")
    print(f"  Safety Margin:      {concurrent['safety_margin'] * 100:.0f}%")
    print(f"  Usable Memory:      {concurrent['usable_memory_gb']:.2f} GB")
    print(f"  Memory per Sweep:   {concurrent['memory_per_sweep_mb']:.1f} MB")
    print(f"  {'─' * 30}")
    print(f"  MAX CONCURRENT:     {concurrent['max_concurrent']}")
    print(f"  Est. Utilization:   {concurrent['estimated_utilization'] * 100:.1f}%")

    print("\n" + "=" * 70)


def reset_config_to_defaults():
    """Reset the global config to defaults."""
    from src.config import (
        Config as ConfigClass,
        GeneralConfig,
        EnvironmentConfig,
        ModelsConfig,
        TrainConfig,
    )
    default = ConfigClass()

    # Reset all fields to defaults (access model_fields from class, not instance)
    for key in GeneralConfig.model_fields:
        setattr(config.general, key, getattr(default.general, key))
    for key in EnvironmentConfig.model_fields:
        setattr(config.environment, key, getattr(default.environment, key))
    for key in ModelsConfig.model_fields:
        setattr(config.models, key, getattr(default.models, key))
    for key in TrainConfig.model_fields:
        setattr(config.train, key, getattr(default.train, key))


def resolve_config_path(cfg_path: str) -> str | None:
    """Resolve config path relative to project root or as absolute."""
    # Try relative to project root first
    full_path = os.path.join(PROJECT_ROOT, cfg_path)
    if os.path.exists(full_path):
        return full_path
    # Try as-is (absolute or cwd-relative)
    if os.path.exists(cfg_path):
        return cfg_path
    return None  # Not found


def run_sweep_profile(config_files: list[str], safety_margin: float = 0.10):
    """Profile multiple config files and show comparison."""
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        print("No GPU available. Cannot profile.")
        return

    print("\n" + "=" * 70)
    print("  MULTI-CONFIG SWEEP PROFILE (CartPole State-Only)")
    print("=" * 70)
    print(f"\nGPU: {gpu_info['device_name']} ({bytes_to_gb(gpu_info['total_memory']):.1f} GB)")
    print(f"Safety Margin: {safety_margin * 100:.0f}%\n")

    results = []
    for i, config_file in enumerate(config_files):
        # Reset config to defaults before loading new file
        reset_config_to_defaults()

        config_path = resolve_config_path(config_file)
        if config_path is None:
            print(f"  Skipping {config_file} (not found)")
            continue

        load_env_config(config_path)

        print(f"Profiling [{i + 1}/{len(config_files)}] {config_file}: "
              f"batch={config.train.batch_size}, "
              f"d_hidden={config.models.d_hidden}, "
              f"pixels={config.general.use_pixels}...")

        result = profile_training_step(config)
        result["config_file"] = config_file
        concurrent = calculate_concurrent_sweeps(
            result["memory"]["peak"],
            gpu_info["total_memory"],
            safety_margin,
        )
        result["concurrent"] = concurrent
        results.append(result)

        # Clear memory between profiles
        reset_memory_stats()

    # Print comparison table
    print(f"\n{'Configuration Comparison':─^70}")
    print(f"{'Config':<35} {'Batch':<6} {'Hidden':<7} {'Peak MB':<9} {'Concurrent':<10}")
    print("─" * 70)

    for r in results:
        cfg = r["config"]
        config_name = os.path.basename(r["config_file"])
        print(f"{config_name:<35} "
              f"{cfg['batch_size']:<6} "
              f"{cfg['d_hidden']:<7} "
              f"{bytes_to_mb(r['memory']['peak']):>7.1f} "
              f"{r['concurrent']['max_concurrent']:>8}")

    # Recommendation
    if results:
        best = max(results, key=lambda x: x["concurrent"]["max_concurrent"])
        print(f"\n{'Recommendation':─^70}")
        print(f"For maximum parallelism, use: {os.path.basename(best['config_file'])}")
        print(f"  → {best['concurrent']['max_concurrent']} concurrent sweeps possible")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU memory for concurrent hyperparameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=str,
        help="Path to env config YAML (e.g., env_configs/cartpole_state_only.yaml)"
    )
    parser.add_argument(
        "--safety_margin", type=float, default=0.10,
        help="Safety margin for GPU memory (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Profile all CartPole state-only configs for comparison"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations before profiling (default: 3)"
    )

    args = parser.parse_args()

    # Check for GPU
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        print("ERROR: No GPU available. This profiler requires CUDA/ROCm.")
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        sys.exit(1)

    if args.sweep:
        # Profile available CartPole configs
        config_files = [
            "env_configs/cartpole_state_only_small.yaml",
        ]
        run_sweep_profile(config_files, args.safety_margin)
    else:
        # Single config profile
        if args.config:
            config_path = resolve_config_path(args.config)
            if config_path is None:
                print(f"ERROR: Config not found: {args.config}")
                sys.exit(1)
            load_env_config(config_path)
            print(f"Loaded config: {args.config}")
        else:
            # Default to state-only for baseline work
            default_config = resolve_config_path("env_configs/cartpole_state_only_small.yaml")
            if default_config:
                load_env_config(default_config)
                print(f"Using default config: env_configs/cartpole_state_only_small.yaml")

        print(f"Profiling: batch={config.train.batch_size}, "
              f"d_hidden={config.models.d_hidden}, "
              f"use_pixels={config.general.use_pixels}...")

        result = profile_training_step(config, warmup_steps=args.warmup)
        concurrent = calculate_concurrent_sweeps(
            result["memory"]["peak"],
            gpu_info["total_memory"],
            args.safety_margin,
        )

        print_profile_report(result, gpu_info, concurrent)


if __name__ == "__main__":
    main()
