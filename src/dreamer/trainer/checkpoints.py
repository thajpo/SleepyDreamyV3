"""Checkpoint saving and loading for DreamerV3 training.

Abstracts file I/O out of the training loop to keep the Trainer stateless.
"""
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    critic_ema,
    q_critic,
    q_critic_ema,
    wm_optimizer,
    actor_optimizer,
    critic_optimizer,
    return_scale,
    ret_lo,
    ret_hi,
    mlflow_run_id=None,
    final=False,
    label=None,
    best_eval_score=None,
    best_eval_step=None,
    best_eval_metric=None,
    run_id=None,
    config_snapshot=None,
) -> str:
    """Save training state and its optional config snapshot atomically."""
    if final and label is not None:
        raise ValueError("final and label are mutually exclusive")
    suffix = "final" if final else label or f"step_{train_step}"
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
        "critic_ema": get_model(critic_ema).state_dict(),
        "q_critic": get_model(q_critic).state_dict(),
        "q_critic_ema": get_model(q_critic_ema).state_dict(),
        "wm_optimizer": wm_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "return_scale": return_scale,
        "ret_lo": ret_lo,
        "ret_hi": ret_hi,
        "mlflow_run_id": mlflow_run_id,
        "best_eval_score": best_eval_score,
        "best_eval_step": best_eval_step,
        "best_eval_metric": best_eval_metric,
        "run_id": run_id,
        "config_snapshot": config_snapshot,
    }
    destination = Path(checkpoint_dir) / f"checkpoint_{suffix}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    path = str(destination)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    checkpoint_path,
    device,
    encoder,
    world_model,
    actor,
    critic,
    critic_ema,
    q_critic,
    q_critic_ema,
    wm_optimizer,
    actor_optimizer,
    critic_optimizer,
    current_return_scale,
    current_ret_lo,
    current_ret_hi,
):
    """
    Load full checkpoint (encoder, world model, actor, critic, optimizers).
    Returns a dictionary of extra state (step, return scales).
    """
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    # Load encoder and world_model
    get_model(encoder).load_state_dict(checkpoint["encoder"])
    get_model(world_model).load_state_dict(checkpoint["world_model"], strict=False)

    # Load actor/critic
    if "actor" in checkpoint:
        get_model(actor).load_state_dict(checkpoint["actor"])
        get_model(critic).load_state_dict(checkpoint["critic"])
        if "critic_ema" in checkpoint:
            get_model(critic_ema).load_state_dict(checkpoint["critic_ema"])
        else:
            # Backward compatibility: old checkpoints did not save critic_ema.
            get_model(critic_ema).load_state_dict(get_model(critic).state_dict())
        if "q_critic" in checkpoint:
            get_model(q_critic).load_state_dict(checkpoint["q_critic"])
            if "q_critic_ema" in checkpoint:
                get_model(q_critic_ema).load_state_dict(checkpoint["q_critic_ema"])
            else:
                get_model(q_critic_ema).load_state_dict(get_model(q_critic).state_dict())

    # Restore optimizers (skip if architecture changed)
    if "wm_optimizer" in checkpoint:
        try:
            wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
        except ValueError as e:
            if "doesn't match the size" in str(e):
                print(f"Warning: Skipping WM optimizer state (architecture changed)")
            else:
                raise
    if "actor_optimizer" in checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if "critic_optimizer" in checkpoint:
        try:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        except ValueError as e:
            if "different number of parameter groups" in str(e) or "size" in str(e):
                print("Warning: Skipping critic optimizer state (architecture changed)")
            else:
                raise

    # Package returns
    step = checkpoint.get("step", 0)
    S = float(checkpoint.get("return_scale", current_return_scale))
    ret_lo = checkpoint.get("ret_lo", current_ret_lo)
    ret_hi = checkpoint.get("ret_hi", current_ret_hi)

    print(f"Resumed checkpoint from {checkpoint_path} at step {step}")

    return {
        "step": step,
        "return_scale": S,
        "ret_lo": ret_lo,
        "ret_hi": ret_hi,
        "best_eval_score": checkpoint.get("best_eval_score"),
        "best_eval_step": checkpoint.get("best_eval_step"),
        "best_eval_metric": checkpoint.get("best_eval_metric"),
        "run_id": checkpoint.get("run_id"),
    }
