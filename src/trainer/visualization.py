"""TensorBoard visualization and logging for DreamerV3 training."""

import torch
import torch.nn.functional as F


def log_metrics(
    writer,
    step,
    total_wm_loss,
    total_actor_loss,
    total_critic_loss,
    wm_loss_components,
    sequence_length,
    dreamed_rewards_list,
    dreamed_values_list,
    actor_entropy_list,
    config,
    log_every=250,
    image_log_every=2500,
    last_obs_pixels=None,
    last_obs_pixels_original=None,
    last_reconstruction_pixels=None,
    last_posterior_probs=None,
    last_dreamed_pixels=None,
):
    """
    Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        step: Current training step
        total_wm_loss: Total world model loss
        total_actor_loss: Total actor loss
        total_critic_loss: Total critic loss
        wm_loss_components: Dict of individual WM loss components
        sequence_length: Sequence length for normalization
        dreamed_rewards_list: List of dreamed reward tensors
        dreamed_values_list: List of dreamed value tensors
        actor_entropy_list: List of actor entropy values
        config: Config object with beta coefficients
        log_every: Steps between scalar logging
        image_log_every: Steps between image logging
        last_obs_pixels: Last observation pixels for visualization
        last_obs_pixels_original: Original resolution pixels
        last_reconstruction_pixels: Reconstructed pixels
        last_posterior_probs: Posterior probabilities
        last_dreamed_pixels: Dreamed pixel sequences
    """
    log_scalars = step % log_every == 0
    log_images = step % image_log_every == 0
    if not (log_scalars or log_images):
        return

    # Safety check - should not happen due to assertions, but just in case
    if (
        total_wm_loss is None
        or total_actor_loss is None
        or total_critic_loss is None
    ):
        return

    if log_scalars:
        _log_scalar_metrics(
            writer,
            step,
            total_wm_loss,
            total_actor_loss,
            total_critic_loss,
            wm_loss_components,
            sequence_length,
            dreamed_rewards_list,
            dreamed_values_list,
            actor_entropy_list,
            config,
        )

    if log_images:
        _log_image_metrics(
            writer,
            step,
            last_obs_pixels,
            last_obs_pixels_original,
            last_reconstruction_pixels,
            last_posterior_probs,
            last_dreamed_pixels,
        )

    writer.flush()


def _log_scalar_metrics(
    writer,
    step,
    total_wm_loss,
    total_actor_loss,
    total_critic_loss,
    wm_loss_components,
    sequence_length,
    dreamed_rewards_list,
    dreamed_values_list,
    actor_entropy_list,
    config,
):
    """Log scalar metrics to TensorBoard."""
    # All losses normalized to per-step for fair comparison
    if sequence_length > 0:
        norm = 1.0 / sequence_length
        beta_pred = config.train.beta_pred
        beta_dyn = config.train.beta_dyn
        beta_rep = config.train.beta_rep

        # Convert tensor loss components to CPU floats (single sync for all 8 components)
        wm_components_cpu = {k: v.item() for k, v in wm_loss_components.items()}

        # Per-step totals
        wm_per_step = total_wm_loss.item() * norm
        writer.add_scalar("loss/world_model/total_per_step", wm_per_step, step)
        writer.add_scalar(
            "loss/actor/total_per_step", total_actor_loss.item() * norm, step
        )
        writer.add_scalar(
            "loss/critic/total_per_step", total_critic_loss.item() * norm, step
        )

        # Raw component values (per-step, unscaled)
        pixel = wm_components_cpu["prediction_pixel"] * norm
        vector = wm_components_cpu["prediction_vector"] * norm
        reward = wm_components_cpu["prediction_reward"] * norm
        cont = wm_components_cpu["prediction_continue"] * norm
        dyn = wm_components_cpu["dynamics"] * norm
        rep = wm_components_cpu["representation"] * norm

        # Prediction sub-components (unscaled, for debugging)
        writer.add_scalar("loss/wm_components/pixel", pixel, step)
        writer.add_scalar("loss/wm_components/vector", vector, step)
        writer.add_scalar("loss/wm_components/reward", reward, step)
        writer.add_scalar("loss/wm_components/continue", cont, step)
        writer.add_scalar("loss/wm_components/dynamics", dyn, step)
        writer.add_scalar("loss/wm_components/representation", rep, step)

        # Scaled contributions to total (these should sum to total_per_step)
        pred_total = pixel + vector + reward + cont
        writer.add_scalar("loss/wm_scaled/prediction", beta_pred * pred_total, step)
        writer.add_scalar("loss/wm_scaled/dynamics", beta_dyn * dyn, step)
        writer.add_scalar("loss/wm_scaled/representation", beta_rep * rep, step)

        # Raw KL divergences (before free bits clipping)
        writer.add_scalar(
            "debug/kl_dynamics_raw",
            wm_components_cpu["kl_dynamics_raw"] * norm,
            step,
        )
        writer.add_scalar(
            "debug/kl_representation_raw",
            wm_components_cpu["kl_representation_raw"] * norm,
            step,
        )
    else:
        # Fallback if no sequence
        writer.add_scalar(
            "loss/world_model/total_per_step", total_wm_loss.item(), step
        )
        writer.add_scalar(
            "loss/actor/total_per_step", total_actor_loss.item(), step
        )
        writer.add_scalar(
            "loss/critic/total_per_step", total_critic_loss.item(), step
        )

    # Dreamed trajectory statistics (debugging metrics)
    # Batch stats computation and sync once to reduce GPU stalls
    if dreamed_rewards_list:
        all_dreamed_rewards = torch.cat(dreamed_rewards_list, dim=0)
        reward_stats = torch.stack(
            [
                all_dreamed_rewards.mean(),
                all_dreamed_rewards.std(),
                all_dreamed_rewards.min(),
                all_dreamed_rewards.max(),
            ]
        ).cpu()  # Single sync for all 4 stats
        writer.add_scalar("debug/dream/reward/mean", reward_stats[0].item(), step)
        writer.add_scalar("debug/dream/reward/std", reward_stats[1].item(), step)
        writer.add_scalar("debug/dream/reward/min", reward_stats[2].item(), step)
        writer.add_scalar("debug/dream/reward/max", reward_stats[3].item(), step)

    if dreamed_values_list:
        all_dreamed_values = torch.cat(dreamed_values_list, dim=0)
        value_stats = torch.stack(
            [
                all_dreamed_values.mean(),
                all_dreamed_values.std(),
            ]
        ).cpu()  # Single sync for both stats
        writer.add_scalar("debug/dream/value/mean", value_stats[0].item(), step)
        writer.add_scalar("debug/dream/value/std", value_stats[1].item(), step)

    # Actor entropy (important for monitoring exploration)
    if actor_entropy_list:
        all_entropy = torch.stack(actor_entropy_list)
        entropy_stats = torch.stack(
            [
                all_entropy.mean(),
                all_entropy.std(),
            ]
        ).cpu()  # Single sync for both stats
        writer.add_scalar("actor/entropy/mean", entropy_stats[0].item(), step)
        writer.add_scalar("actor/entropy/std", entropy_stats[1].item(), step)


def _log_image_metrics(
    writer,
    step,
    last_obs_pixels,
    last_obs_pixels_original,
    last_reconstruction_pixels,
    last_posterior_probs,
    last_dreamed_pixels,
):
    """Log image metrics to TensorBoard."""
    if last_obs_pixels is None or last_reconstruction_pixels is None:
        return

    # Take first sample from batch
    actual = (last_obs_pixels[0] / 255.0).clamp(0, 1)  # (C, H, W)
    recon = torch.sigmoid(last_reconstruction_pixels[0]).clamp(0, 1)  # (C, H, W)

    # Resize reconstruction to match actual if needed
    if recon.shape != actual.shape:
        recon = F.interpolate(
            recon.unsqueeze(0),
            size=actual.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # 1. Actual vs Reconstruction side by side
    comparison = torch.cat([actual, recon], dim=2)  # concat on width
    writer.add_image("images/actual_vs_reconstruction", comparison, step)

    # 2. Reconstruction error heatmap (absolute diff, averaged over RGB)
    error = torch.abs(actual - recon).mean(dim=0, keepdim=True)  # (1, H, W)
    error_norm = error / (error.max() + 1e-8)
    error_heatmap = error_norm.repeat(3, 1, 1)  # grayscale to RGB
    writer.add_image("images/reconstruction_error", error_heatmap, step)

    # 3. Latent activation heatmap
    if last_posterior_probs is not None:
        # Shape: (batch, d_hidden, categories) -> take first batch, make 2D
        latent_probs = last_posterior_probs[0]  # (512, 32)
        # Normalize and add batch/channel dims for add_images
        latent_img = latent_probs.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 32)
        writer.add_images("images/latent_activations", latent_img, step)

    # 4. Dream rollout video (imagined future frames)
    if last_dreamed_pixels is not None:
        # Shape: (n_dream_steps, batch, C, H, W) -> take first batch
        dream_frames = last_dreamed_pixels[:, 0]  # (n_steps, C, H, W)
        # Resize to match actual size if needed
        if dream_frames.shape[2:] != actual.shape[1:]:
            dream_frames = F.interpolate(
                dream_frames,
                size=actual.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
        # Add video: shape (N, T, C, H, W) - batch, time, channels, height, width
        video = dream_frames.unsqueeze(0)  # (1, n_steps, C, H, W)
        writer.add_video("video/dream_rollout", video, step, fps=4)
        # Also add strip image for quick glance
        n_show = min(5, dream_frames.shape[0])
        dream_strip = torch.cat([dream_frames[i] for i in range(n_show)], dim=2)
        writer.add_images("images/dream_rollout", dream_strip.unsqueeze(0), step)

    # 5. Original resolution image (larger, easier to see)
    if last_obs_pixels_original is not None:
        original = (last_obs_pixels_original[0] / 255.0).clamp(0, 1)  # First sample only
        writer.add_image("images/original_resolution", original, step)
