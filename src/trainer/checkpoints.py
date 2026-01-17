"""Checkpoint saving and loading for DreamerV3 training."""

import os
import torch


def get_model(model):
    """Get underlying model (handles both compiled and non-compiled models)."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    checkpoint_dir,
    train_step,
    encoder,
    world_model,
    actor,
    critic,
    wm_optimizer,
    actor_optimizer,
    critic_optimizer,
    final=False,
):
    """
    Save all model checkpoints.

    Args:
        checkpoint_dir: Directory to save checkpoints
        train_step: Current training step
        encoder: Encoder network
        world_model: World model network
        actor: Actor network
        critic: Critic network
        wm_optimizer: World model optimizer
        actor_optimizer: Actor optimizer
        critic_optimizer: Critic optimizer
        final: If True, save as final checkpoint

    Returns:
        Path to saved checkpoint
    """
    suffix = "final" if final else f"step_{train_step}"
    checkpoint = {
        "step": train_step,
        "encoder": get_model(encoder).state_dict(),
        "world_model": {
            k: v
            for k, v in get_model(world_model).state_dict().items()
            if k not in ("h_prev", "z_prev")
        },
        "actor": get_model(actor).state_dict(),
        "critic": get_model(critic).state_dict(),
        "wm_optimizer": wm_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}.pt")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    return path


def save_wm_only_checkpoint(
    checkpoint_dir,
    train_step,
    encoder,
    world_model,
    wm_optimizer,
    final=False,
):
    """
    Save WM-only checkpoint for bootstrap phase (no actor/critic).

    Args:
        checkpoint_dir: Directory to save checkpoints
        train_step: Current training step
        encoder: Encoder network
        world_model: World model network
        wm_optimizer: World model optimizer
        final: If True, save as final checkpoint

    Returns:
        Path to saved checkpoint
    """
    suffix = "final" if final else f"step_{train_step}"
    checkpoint = {
        "step": train_step,
        "encoder": get_model(encoder).state_dict(),
        "world_model": {
            k: v
            for k, v in get_model(world_model).state_dict().items()
            if k not in ("h_prev", "z_prev")
        },
        "wm_optimizer": wm_optimizer.state_dict(),
        "checkpoint_type": "wm_only",
    }
    path = os.path.join(checkpoint_dir, f"wm_checkpoint_{suffix}.pt")
    torch.save(checkpoint, path)
    print(f"WM-only checkpoint saved: {path}")
    return path


def load_checkpoint(
    checkpoint_path,
    device,
    encoder,
    world_model,
    actor,
    critic,
    wm_optimizer,
    actor_optimizer,
    critic_optimizer,
    reset_ac=False,
):
    """
    Load checkpoint with explicit control over actor/critic loading.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device
        encoder: Encoder network
        world_model: World model network
        actor: Actor network
        critic: Critic network
        wm_optimizer: World model optimizer
        actor_optimizer: Actor optimizer
        critic_optimizer: Critic optimizer
        reset_ac: If True, skip loading actor/critic (keep random init)

    Returns:
        Tuple of (checkpoint_type, train_step) where:
            - checkpoint_type: 'wm_only', 'full', or 'reset_ac'
            - train_step: Step to resume from (0 if reset_ac or wm_only)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load encoder and world_model (always present)
    get_model(encoder).load_state_dict(checkpoint["encoder"])
    get_model(world_model).load_state_dict(
        checkpoint["world_model"], strict=False
    )

    # Restore WM optimizer if present
    if "wm_optimizer" in checkpoint:
        wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])

    # Handle actor/critic based on explicit user intent
    has_ac = "actor" in checkpoint

    if reset_ac:
        # User explicitly requested fresh actor/critic
        print(f"Loaded WM weights from {checkpoint_path}")
        print("Actor/critic reset to random (--reset-ac)")
        return "reset_ac", 0
    elif not has_ac:
        # WM-only checkpoint, no AC to load
        print(f"Loaded WM-only checkpoint from {checkpoint_path}")
        print("Actor/critic initialized randomly")
        return "wm_only", 0
    else:
        # Full checkpoint with --resume: load everything
        get_model(actor).load_state_dict(checkpoint["actor"])
        get_model(critic).load_state_dict(checkpoint["critic"])

        if "actor_optimizer" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        train_step = checkpoint.get("step", 0)
        print(
            f"Resumed full checkpoint from {checkpoint_path} at step {train_step}"
        )
        return "full", train_step
