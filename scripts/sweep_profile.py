#!/usr/bin/env python3
"""
sweep_profile.py - GPU Memory Profiler for Concurrent Hyperparameter Sweeps

Profiles GPU memory usage during training to determine how many concurrent
sweeps can safely run on your hardware.

Usage:
    python scripts/sweep_profile.py                    # Profile default config
    python scripts/sweep_profile.py --batch_size 32    # Override batch size
    python scripts/sweep_profile.py --d_hidden 128     # Override hidden dim
    python scripts/sweep_profile.py --sweep            # Profile multiple configs
    python scripts/sweep_profile.py --safety_margin 0.15  # 15% safety margin

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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.config import Config, get_default_device
from src.trainer_utils import (
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    symlog,
    twohot_encode,
)


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def bytes_to_gb(b: int) -> float:
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


def create_dummy_batch(config: Config, device: torch.device) -> dict:
    """Create a dummy batch for profiling."""
    batch_size = config.train.batch_size
    seq_len = config.train.sequence_length
    n_obs = config.environment.n_observations
    n_actions = config.environment.n_actions

    batch = {
        "state": torch.randn(batch_size, seq_len, n_obs, device=device),
        "action": F.one_hot(
            torch.randint(0, n_actions, (batch_size, seq_len), device=device),
            num_classes=n_actions,
        ).float(),
        "reward": torch.randn(batch_size, seq_len, device=device),
        "done": torch.zeros(batch_size, seq_len, device=device),
    }

    if config.general.use_pixels:
        target_size = config.models.encoder.cnn.target_size
        batch["pixels"] = torch.randint(
            0, 256,
            (batch_size, seq_len, 3, target_size[0], target_size[1]),
            device=device,
            dtype=torch.float32,
        )

    return batch


def profile_training_step(config: Config, warmup_steps: int = 3) -> dict:
    """
    Profile a single training step and measure peak memory usage.

    Returns dict with memory stats and parameter counts.
    """
    device = torch.device(config.general.device)

    # Reset memory tracking
    reset_memory_stats()

    # Initialize models
    encoder, world_model = initialize_world_model(
        device, config, batch_size=config.train.batch_size
    )
    actor = initialize_actor(device, config)
    critic = initialize_critic(device, config)

    # Count parameters
    encoder_params = count_parameters(encoder)
    world_model_params = count_parameters(world_model)
    actor_params = count_parameters(actor)
    critic_params = count_parameters(critic)
    total_params = encoder_params + world_model_params + actor_params + critic_params

    # Initialize optimizers (Adam stores 2 states per param: momentum & variance)
    wm_params = list(encoder.parameters()) + list(world_model.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=config.train.wm_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.train.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.train.critic_lr)

    # Memory after model + optimizer initialization
    model_init_memory = get_memory_stats()

    # Create reward bins for loss computation
    b_start = config.train.b_start
    b_end = config.train.b_end
    B = torch.arange(b_start, b_end, device=device).float()
    B = torch.sign(B) * (torch.exp(torch.abs(B)) - 1)  # symexp

    # Warmup passes (for torch.compile and CUDA kernel caching)
    batch = create_dummy_batch(config, device)

    for _ in range(warmup_steps):
        # Simulate a forward pass
        _run_forward_pass(encoder, world_model, actor, critic, batch, config, B, device)
        reset_memory_stats()

    # Profile actual training step
    reset_memory_stats()

    # Forward pass
    loss, actor_loss, critic_loss = _run_forward_pass(
        encoder, world_model, actor, critic, batch, config, B, device
    )
    forward_memory = get_memory_stats()

    # Backward pass - world model
    loss.backward()
    backward_wm_memory = get_memory_stats()

    # Backward pass - actor/critic
    if actor_loss is not None:
        actor_loss.backward()
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
            "batch_size": config.train.batch_size,
            "d_hidden": config.models.d_hidden,
            "seq_length": config.train.sequence_length,
            "n_gru_blocks": config.models.rnn.n_blocks,
            "use_pixels": config.general.use_pixels,
            "n_dream_steps": config.train.num_dream_steps,
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


def _run_forward_pass(encoder, world_model, actor, critic, batch, config, B, device):
    """Run a forward pass simulating training."""
    batch_size = config.train.batch_size
    seq_len = config.train.sequence_length
    n_actions = config.environment.n_actions
    n_dream_steps = config.train.num_dream_steps

    # Reset world model hidden state
    world_model.h_prev = torch.zeros_like(world_model.h_prev)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    all_h_z = []

    # Process sequence
    for t in range(seq_len):
        if config.general.use_pixels:
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
        obs_rec, reward_logits, continue_logits, h_z, prior_logits = world_model(
            posterior_dist, action
        )

        all_h_z.append(h_z)

        # Compute losses
        if config.general.use_pixels:
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
        z_flat = h_z_dream[:, world_model.d_hidden * config.models.rnn.n_blocks:]
        z_embed = world_model.z_embedding(z_flat)
        h_dream, prior_logits = world_model.step_dynamics(z_embed, action_onehot, h_dream)

        # Sample from prior for next state
        prior_dist = Categorical(logits=prior_logits)
        z_idx = prior_dist.sample()
        z_onehot = F.one_hot(z_idx, num_classes=config.models.d_hidden // 16).float()
        z_flat_new = z_onehot.view(batch_size, -1)
        h_z_dream = torch.cat([h_dream, z_flat_new], dim=-1)

    # Simple actor/critic loss (not full implementation, just for memory profiling)
    if dream_values:
        values = torch.stack([v.mean() for v in dream_values])
        log_probs = torch.stack(dream_log_probs)
        actor_loss = -(log_probs * values.detach()).mean()
        critic_loss = F.mse_loss(values, torch.zeros_like(values))
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


def run_sweep_profile(configs: list[dict], safety_margin: float = 0.10):
    """Profile multiple configurations and show comparison."""
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        print("No GPU available. Cannot profile.")
        return

    print("\n" + "=" * 70)
    print("  MULTI-CONFIG SWEEP PROFILE")
    print("=" * 70)
    print(f"\nGPU: {gpu_info['device_name']} ({bytes_to_gb(gpu_info['total_memory']):.1f} GB)")
    print(f"Safety Margin: {safety_margin * 100:.0f}%\n")

    results = []
    for i, cfg_override in enumerate(configs):
        # Create config with overrides
        config = Config()
        if "batch_size" in cfg_override:
            config.train.batch_size = cfg_override["batch_size"]
        if "d_hidden" in cfg_override:
            config.models.d_hidden = cfg_override["d_hidden"]
        if "use_pixels" in cfg_override:
            config.general.use_pixels = cfg_override["use_pixels"]

        print(f"Profiling config {i + 1}/{len(configs)}: "
              f"batch={config.train.batch_size}, "
              f"d_hidden={config.models.d_hidden}, "
              f"pixels={config.general.use_pixels}...")

        result = profile_training_step(config)
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
    print(f"{'Batch':<8} {'Hidden':<8} {'Pixels':<8} {'Params':<12} {'Peak MB':<10} {'Concurrent':<10}")
    print("─" * 70)

    for r in results:
        cfg = r["config"]
        print(f"{cfg['batch_size']:<8} "
              f"{cfg['d_hidden']:<8} "
              f"{str(cfg['use_pixels']):<8} "
              f"{r['params']['total']:>10,} "
              f"{bytes_to_mb(r['memory']['peak']):>8.1f} "
              f"{r['concurrent']['max_concurrent']:>8}")

    # Recommendation
    best = max(results, key=lambda x: x["concurrent"]["max_concurrent"])
    print(f"\n{'Recommendation':─^70}")
    print(f"For maximum parallelism, use:")
    print(f"  batch_size={best['config']['batch_size']}, "
          f"d_hidden={best['config']['d_hidden']}, "
          f"use_pixels={best['config']['use_pixels']}")
    print(f"  → {best['concurrent']['max_concurrent']} concurrent sweeps")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU memory for concurrent hyperparameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size (default: from config)"
    )
    parser.add_argument(
        "--d_hidden", type=int, default=None,
        help="Override hidden dimension (default: 256)"
    )
    parser.add_argument(
        "--use_pixels", action="store_true", default=None,
        help="Enable pixel observations"
    )
    parser.add_argument(
        "--state_only", action="store_true",
        help="Disable pixel observations (state-only)"
    )
    parser.add_argument(
        "--safety_margin", type=float, default=0.10,
        help="Safety margin for GPU memory (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Profile multiple configurations for comparison"
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
        # Profile multiple configurations
        configs = [
            # Batch size variations (state-only for fast profiling)
            {"batch_size": 8, "d_hidden": 256, "use_pixels": False},
            {"batch_size": 16, "d_hidden": 256, "use_pixels": False},
            {"batch_size": 32, "d_hidden": 256, "use_pixels": False},
            {"batch_size": 64, "d_hidden": 256, "use_pixels": False},
            # Hidden dim variations
            {"batch_size": 16, "d_hidden": 64, "use_pixels": False},
            {"batch_size": 16, "d_hidden": 128, "use_pixels": False},
            {"batch_size": 16, "d_hidden": 256, "use_pixels": False},
            # With pixels (higher memory)
            {"batch_size": 8, "d_hidden": 256, "use_pixels": True},
            {"batch_size": 16, "d_hidden": 256, "use_pixels": True},
        ]
        run_sweep_profile(configs, args.safety_margin)
    else:
        # Single config profile
        config = Config()

        # Apply overrides
        if args.batch_size is not None:
            config.train.batch_size = args.batch_size
        if args.d_hidden is not None:
            config.models.d_hidden = args.d_hidden
        if args.use_pixels:
            config.general.use_pixels = True
        if args.state_only:
            config.general.use_pixels = False

        print(f"Profiling with batch_size={config.train.batch_size}, "
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
