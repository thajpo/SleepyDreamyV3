"""Visualize world model dreams as MP4 videos."""

import argparse
import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..models import (
    initialize_actor,
    initialize_world_model,
    symlog,
    resize_pixels_to_target,
)
from ..envs.utils import create_env


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
    target_size = config.models.encoder.cnn.target_size

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
        z_onehot = F.one_hot(
            z_indices, num_classes=config.models.d_hidden // 16
        ).float()
        z_indices = posterior_dist.sample()
        z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())

        # Get z embedding
        z_flat = z_sample.view(1, -1)
        z_embed = world_model.z_embedding(z_flat)

        # Initialize h and step dynamics once to get initial state
        h = torch.zeros(
            1, config.models.d_hidden * config.models.rnn.n_blocks, device=device
        )
        action_zero = torch.zeros(1, config.environment.n_actions, device=device)
        h, _ = world_model.step_dynamics(z_embed, action_zero, h)

    return h, z_sample, z_embed, pixels


def dream_rollout(
    config, world_model, actor, h_init, z_embed_init, num_steps, device, use_actor=False
):
    """
    Roll out the world model for num_steps, returning decoded frames.

    Returns:
        frames: List of decoded pixel tensors (num_steps, C, H, W)
        actions: List of actions taken
    """
    frames = []
    actions = []

    h = h_init.clone()
    z_embed = z_embed_init.clone()
    n_actions = config.environment.n_actions
    d_hidden = config.models.d_hidden

    with torch.no_grad():
        for step in range(num_steps):
            # Decode current state to pixels
            # First need to get z_sample from z_embed (approximate by using prior)
            _, prior_logits = world_model.step_dynamics(
                z_embed, torch.zeros(1, n_actions, device=device), h
            )
            prior_dist = dist.Categorical(logits=prior_logits, validate_args=False)
            z_indices = prior_dist.sample()
            z_sample = F.one_hot(z_indices, num_classes=d_hidden // 16).float()

            # Form full state for decoder
            h_z = world_model.join_h_and_z(h, z_sample)

            # Decode to pixels
            decoded = world_model.decoder(h_z)
            pixels = torch.sigmoid(decoded["pixels"])  # (1, C, H, W)
            frames.append(pixels.squeeze(0).cpu())

            # Select action
            if use_actor and actor is not None:
                action_logits = actor(h_z)
                action_dist = dist.Categorical(
                    logits=action_logits, validate_args=False
                )
                action = action_dist.sample()
            else:
                action = torch.randint(0, n_actions, (1,), device=device)

            actions.append(action.item())
            action_onehot = F.one_hot(action, num_classes=n_actions).float()

            # Step dynamics to get next state
            h, prior_logits = world_model.step_dynamics(z_embed, action_onehot, h)

            # Sample next z from prior
            prior_dist = dist.Categorical(logits=prior_logits, validate_args=False)
            z_indices = prior_dist.sample()
            z_sample = F.one_hot(z_indices, num_classes=d_hidden // 16).float()
            z_embed = world_model.z_embedding(z_sample.view(1, -1))

    return frames, actions


def frames_to_video(frames, output_path, fps=10, initial_frame=None):
    """Save frames as MP4 video using imageio."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio not found. Install with: pip install imageio[ffmpeg]")
        return False

    video_frames = []

    # Add initial frame if provided (real observation)
    if initial_frame is not None:
        # initial_frame is (C, H, W) tensor in [0, 1]
        frame_np = (initial_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Add border to distinguish from dreams
        frame_np = add_border(frame_np, color=(0, 255, 0), thickness=3)  # Green = real
        video_frames.append(frame_np)

    # Add dream frames
    for i, frame in enumerate(frames):
        # frame is (C, H, W) tensor in [0, 1]
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Add border to distinguish dreams
        frame_np = add_border(frame_np, color=(255, 0, 0), thickness=3)  # Red = dream
        video_frames.append(frame_np)

    # Write video
    iio.imwrite(output_path, video_frames, fps=fps, codec="libx264")
    print(f"Saved video: {output_path}")
    return True


def add_border(frame, color, thickness):
    """Add a colored border to a frame."""
    h, w, c = frame.shape
    bordered = frame.copy()
    bordered[:thickness, :] = color  # Top
    bordered[-thickness:, :] = color  # Bottom
    bordered[:, :thickness] = color  # Left
    bordered[:, -thickness:] = color  # Right
    return bordered


def create_comparison_grid(initial_frame, dream_frames, grid_cols=6):
    """Create a grid showing initial frame + dream sequence."""
    # Initial frame with green border, dreams with red border
    all_frames = []

    # Add initial (real) frame
    init_np = (initial_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    init_np = add_border(init_np, color=(0, 255, 0), thickness=3)
    all_frames.append(init_np)

    # Add dream frames
    for frame in dream_frames:
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_np = add_border(frame_np, color=(255, 0, 0), thickness=3)
        all_frames.append(frame_np)

    # Pad to fill grid
    h, w, c = all_frames[0].shape
    while len(all_frames) % grid_cols != 0:
        all_frames.append(np.zeros((h, w, c), dtype=np.uint8))

    # Build grid
    rows = []
    for i in range(0, len(all_frames), grid_cols):
        row = np.concatenate(all_frames[i : i + grid_cols], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)

    return grid


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup
    device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Environment: {cfg.environment.environment_name}")
    print(f"d_hidden: {cfg.models.d_hidden}")
    use_pixels = cfg.general.use_pixels
    print(f"use_pixels: {use_pixels}")

    checkpoint_path = cfg.get("checkpoint", None)
    if not checkpoint_path:
        print("Error: No checkpoint specified. Use checkpoint=/path/to/model.pt")
        return

    num_steps = cfg.get("steps", 30)
    num_rollouts = cfg.get("num_rollouts", 1)
    use_actor = cfg.get("use_actor", False)
    output_prefix = cfg.get("output", "dream")
    fps = cfg.get("fps", 10)

    # Load models
    encoder, world_model, actor = load_models(
        cfg, checkpoint_path, device, load_actor=use_actor
    )

    # Create environment and get initial observation
    env = create_env(cfg.environment.environment_name, use_pixels=use_pixels)
    obs, info = env.reset()
    env.close()

    # Encode initial observation
    h_init, z_sample, z_embed_init, initial_pixels = encode_observation(
        cfg, obs, encoder, world_model, device
    )
    initial_frame = (initial_pixels.squeeze(0) / 255.0).cpu()  # Normalize to [0,1]

    print(f"\nGenerating {num_rollouts} dream rollout(s) of {num_steps} steps...")

    output_dir = Path(".")

    for rollout_idx in range(num_rollouts):
        # Generate dream rollout
        frames, actions = dream_rollout(
            cfg,
            world_model,
            actor,
            h_init,
            z_embed_init,
            num_steps,
            device,
            use_actor=use_actor,
        )

        # Build output filename
        suffix = f"_{rollout_idx}" if num_rollouts > 1 else ""
        video_path = output_dir / f"{output_prefix}{suffix}.mp4"

        # Save video
        frames_to_video(frames, str(video_path), fps=fps, initial_frame=initial_frame)

        # Print action sequence
        action_str = "".join(str(a) for a in actions[:20])
        if len(actions) > 20:
            action_str += "..."
        print(f"  Rollout {rollout_idx}: actions = [{action_str}]")

    print("\nDone!")
    print("Legend: Green border = real observation, Red border = dreamed")


if __name__ == "__main__":
    main()
