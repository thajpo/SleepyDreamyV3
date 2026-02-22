"""Visualize world model dreams as MP4 videos."""

import argparse
import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from pathlib import Path

from .models import (
    initialize_actor,
    initialize_world_model,
    symlog,
    resize_pixels_to_target,
)
from .runtime.env import create_env


def load_models(config, checkpoint_path, device, load_actor=False):
    """Load encoder, world model, and optionally actor from checkpoint."""
    encoder, world_model = initialize_world_model(device, batch_size=1, cfg=config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"], strict=False)

    encoder.eval()
    world_model.eval()

    actor = None
    if load_actor and "actor" in checkpoint:
        actor = initialize_actor(device, cfg=config)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        print("Actor loaded from checkpoint")
    elif load_actor:
        print(
            "Warning: --use-actor specified but no actor in checkpoint, using random actions"
        )

    step = checkpoint.get("step", "unknown")
    print(f"Loaded checkpoint from step {step}")

    return encoder, world_model, actor


def encode_observation(config, obs, encoder, world_model, device):
    """Encode an observation to get (h, z) state."""
    target_size = (64, 64)

    # Prepare pixel observation
    pixels = torch.from_numpy(obs["pixels"]).to(device).float()
    pixels = pixels.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Resize if needed
    if pixels.shape[-2:] != target_size:
        pixels = resize_pixels_to_target(pixels, target_size)

    # Prepare state vector
    state = torch.from_numpy(obs["state"]).to(device).float().unsqueeze(0)
    state = symlog(state)

    obs_dict = {"pixels": pixels, "state": state}

    with torch.no_grad():
        # Encode to get posterior
        posterior_logits = encoder(obs_dict)
        posterior_dist = dist.Categorical(logits=posterior_logits, validate_args=False)

        # Sample z
        z_indices = posterior_dist.sample()
        z_onehot = F.one_hot(z_indices, num_classes=config.d_hidden // 16).float()
        z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())

        # Get z embedding
        z_flat = z_sample.view(1, -1)
        z_embed = world_model.z_embedding(z_flat)

        # Initialize h and step dynamics once to get initial state
        h = torch.zeros(1, config.d_hidden * 4, device=device)
        action_zero = torch.zeros(1, 2, device=device)
        h, _ = world_model.step_dynamics(z_embed, action_zero, h)

    return h, z_sample, z_embed, pixels


def dream_rollout(
    world_model,
    actor,
    h_init,
    z_embed_init,
    num_steps,
    device,
    use_actor=False,
):
    """
    Roll out the world model for num_steps, returning decoded frames.

    Returns:
        List of decoded frames (numpy arrays)
    """
    frames = []
    action_np = 0

    # Rollout loop
    for step in range(num_steps):
        # Get action
        if use_actor and actor is not None:
            actor_input = world_model.join_h_and_z(h_init, z_embed_init)
            action_dist = dist.Categorical(logits=actor(actor_input))
            action = action_dist.sample()
            action_np = action.item()
        else:
            action_np = 0
            action = torch.tensor(action_np, device=device)

        # One-hot action
        action_onehot = F.one_hot(action, num_classes=2).float().unsqueeze(0)

        # Step world model
        _, (h_next, z_next), _ = world_model.step(
            action_onehot.unsqueeze(0), (h_init, z_embed_init)
        )
        h_init = h_next
        z_embed_init = z_next

        # Store frame
        frames.append(action_np)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--use_actor", action="store_true")
    parser.add_argument("--output", default="dream")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # Load dummy config (will be replaced with argparse in future)
    class DummyConfig:
        d_hidden = 128
        num_latents = 32
        n_actions = 2
        n_observations = 4
        use_pixels = False

    config = DummyConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    encoder, world_model, actor = load_models(
        config, args.checkpoint, device, load_actor=args.use_actor
    )

    print("Visualization ready to run (not functional yet)")
    # TODO: Complete rollout and video saving logic


if __name__ == "__main__":
    main()
