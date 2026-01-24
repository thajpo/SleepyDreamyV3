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
    mlflow_run_id=None,
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
        mlflow_run_id: MLflow run ID for resume support

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
        "mlflow_run_id": mlflow_run_id,
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}.pt")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
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
):
    """
    Load full checkpoint (encoder, world model, actor, critic, optimizers).

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

    Returns:
        train_step: Step to resume from
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load encoder and world_model
    get_model(encoder).load_state_dict(checkpoint["encoder"])
    get_model(world_model).load_state_dict(
        checkpoint["world_model"], strict=False
    )

    # Load actor/critic if present
    if "actor" in checkpoint:
        get_model(actor).load_state_dict(checkpoint["actor"])
        get_model(critic).load_state_dict(checkpoint["critic"])

    # Restore optimizers if present
    if "wm_optimizer" in checkpoint:
        wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
    if "actor_optimizer" in checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if "critic_optimizer" in checkpoint:
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    train_step = checkpoint.get("step", 0)
    print(
        f"Resumed checkpoint from {checkpoint_path} at step {train_step}"
    )
    return train_step
