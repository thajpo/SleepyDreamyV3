"""Logging and metrics for DreamerV3 training.

All per-batch accumulator state lives in StepMetrics. Created at the start
of each training step, mutated during the inner loop, consumed at the end
by log_step_metrics().
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ..models.math_utils import symlog


@dataclass
class StepMetrics:
    """All per-batch accumulator state for logging.

    Created once per training step. The inner loop appends to the lists;
    log_step_metrics() consumes everything at the end.
    """

    wm_components: dict[str, torch.Tensor] = field(default_factory=dict)
    dreamed_rewards: list[torch.Tensor] = field(default_factory=list)
    dreamed_values: list[torch.Tensor] = field(default_factory=list)
    actor_entropy: list[torch.Tensor] = field(default_factory=list)
    replay_posterior_states: list[torch.Tensor] = field(default_factory=list)
    replay_value_annotations: list[torch.Tensor] = field(default_factory=list)
    replay_loss: Optional[torch.Tensor] = None
    replay_ema_reg: Optional[torch.Tensor] = None
    viz_data: Optional[dict[str, torch.Tensor]] = None


def create_step_metrics(device: torch.device, do_log_images: bool) -> StepMetrics:
    """Initialize all per-batch logging accumulators in one place."""
    wm_components = {
        "prediction_pixel": torch.tensor(0.0, device=device),
        "prediction_vector": torch.tensor(0.0, device=device),
        "prediction_reward": torch.tensor(0.0, device=device),
        "prediction_continue": torch.tensor(0.0, device=device),
        "dynamics": torch.tensor(0.0, device=device),
        "representation": torch.tensor(0.0, device=device),
        "kl_dynamics_raw": torch.tensor(0.0, device=device),
        "kl_representation_raw": torch.tensor(0.0, device=device),
    }
    return StepMetrics(
        wm_components=wm_components,
        viz_data={} if do_log_images else None,
    )


def collect_viz_data(
    metrics: StepMetrics,
    t_step: int,
    T: int,
    obs_t: dict,
    obs_reconstruction: dict,
    posterior_logits: torch.Tensor,
    batch,
    use_pixels: bool,
):
    """Collect visualization data at the last timestep of the batch."""
    if metrics.viz_data is None or t_step != T - 1:
        return
    if use_pixels and "pixels" in obs_t:
        metrics.viz_data["obs_pixels"] = obs_t["pixels"]
        metrics.viz_data["obs_pixels_original"] = batch.pixels_original[:, t_step]
        metrics.viz_data["reconstruction_pixels"] = obs_reconstruction.get("pixels")
    metrics.viz_data["posterior_probs"] = F.softmax(posterior_logits, dim=-1).detach()


def log_step_metrics(
    logger,
    metrics: StepMetrics,
    total_wm_loss,
    total_actor_loss,
    total_critic_loss,
    sequence_length: int,
    step: int,
    config,
    has_pixel_obs: bool,
    has_vector_obs: bool,
    log_every: int,
    image_log_every: int,
    log_profile: str,
    wm_optimizer=None,
    actor_optimizer=None,
    critic_optimizer=None,
    wm_ac_ratio_cosine: bool = False,
    get_current_wm_ac_ratio=None,
):
    """Log scalar metrics and images to MLflow. Consumes StepMetrics."""
    log_scalars = step % log_every == 0
    do_log_images = step % image_log_every == 0
    if not (log_scalars or do_log_images):
        return

    if total_wm_loss is None or total_actor_loss is None or total_critic_loss is None:
        return

    if log_scalars:
        is_full = log_profile == "full"
        m: dict[str, float] = {}

        if sequence_length > 0:
            norm = 1.0 / sequence_length
            wm_cpu = {k: v.item() for k, v in metrics.wm_components.items()}

            m["loss/wm/total"] = total_wm_loss.item() * norm
            m["loss/actor/total"] = total_actor_loss.item() * norm
            m["loss/critic/total"] = total_critic_loss.item() * norm

            if metrics.replay_loss is not None:
                m["loss/critic/replay"] = float(metrics.replay_loss.item())
            if metrics.replay_ema_reg is not None:
                m["loss/critic/replay_ema_reg"] = float(metrics.replay_ema_reg.item())

            pixel = wm_cpu["prediction_pixel"] * norm
            state = wm_cpu["prediction_vector"] * norm
            reward = wm_cpu["prediction_reward"] * norm
            cont = wm_cpu["prediction_continue"] * norm
            dyn = wm_cpu["dynamics"] * norm
            rep = wm_cpu["representation"] * norm

            if has_pixel_obs:
                m["wm/decoder/pixel_loss"] = pixel
            if has_vector_obs:
                m["wm/decoder/state_loss"] = state

            m["wm/reward_head/loss"] = reward
            m["wm/continue_head/loss"] = cont
            m["wm/rssm/kl_dynamics"] = dyn
            m["wm/rssm/kl_representation"] = rep

            if is_full:
                m["wm/scaled/prediction"] = config.beta_pred * (pixel + state + reward + cont)
                m["wm/scaled/dynamics"] = config.beta_dyn * dyn
                m["wm/scaled/representation"] = config.beta_rep * rep
                m["wm/rssm/kl_dynamics_raw"] = wm_cpu["kl_dynamics_raw"] * norm
                m["wm/rssm/kl_representation_raw"] = wm_cpu["kl_representation_raw"] * norm
        else:
            m["loss/wm/total"] = total_wm_loss.item()
            m["loss/actor/total"] = total_actor_loss.item()
            m["loss/critic/total"] = total_critic_loss.item()

        if metrics.dreamed_rewards:
            all_dr = torch.cat(metrics.dreamed_rewards, dim=0)
            m["dream/wm_reward/mean"] = all_dr.mean().item()
            m["dream/wm_reward/std"] = all_dr.std().item()
            if is_full:
                m["dream/wm_reward/min"] = all_dr.min().item()
                m["dream/wm_reward/max"] = all_dr.max().item()

        if metrics.dreamed_values:
            all_dv = torch.cat(metrics.dreamed_values, dim=0)
            m["dream/critic_value/mean"] = all_dv.mean().item()
            m["dream/critic_value/std"] = all_dv.std().item()
            sv = symlog(all_dv)
            m["dream/critic_value_symlog/mean"] = sv.mean().item()
            m["dream/critic_value_symlog/std"] = sv.std().item()

        if metrics.actor_entropy:
            ae = torch.stack(metrics.actor_entropy)
            m["actor/entropy/mean"] = ae.mean().item()
            if is_full:
                m["actor/entropy/std"] = ae.std().item()

        if is_full:
            if wm_optimizer:
                m["train/lr/wm"] = wm_optimizer.param_groups[0]["lr"]
            if actor_optimizer:
                m["train/lr/actor"] = actor_optimizer.param_groups[0]["lr"]
            if critic_optimizer:
                m["train/lr/critic"] = critic_optimizer.param_groups[0]["lr"]
            if wm_ac_ratio_cosine and get_current_wm_ac_ratio:
                m["train/wm_ac_ratio"] = get_current_wm_ac_ratio()

        logger.log_scalars(m, step)

    # --- Image / video logging ---
    if do_log_images and metrics.viz_data:
        _log_viz(logger, metrics.viz_data, step)


def _log_viz(logger, viz_data: dict, step: int):
    """Log reconstruction images, latent posteriors, and dream videos."""
    obs_pixels = viz_data.get("obs_pixels")
    recon_pixels = viz_data.get("reconstruction_pixels")
    posterior_probs = viz_data.get("posterior_probs")
    dreamed_pixels = viz_data.get("dreamed_pixels")
    obs_pixels_original = viz_data.get("obs_pixels_original")

    if obs_pixels is not None and recon_pixels is not None:
        actual = (obs_pixels[0] / 255.0).clamp(0, 1)
        recon = torch.sigmoid(recon_pixels[0]).clamp(0, 1)
        if recon.shape != actual.shape:
            recon = F.interpolate(
                recon.unsqueeze(0),
                size=actual.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        logger.log_image("viz/decoder/reconstruction", torch.cat([actual, recon], dim=2), step)

        error = torch.abs(actual - recon).mean(dim=0, keepdim=True)
        error_norm = error / (error.max() + 1e-8)
        logger.log_image("viz/decoder/error", error_norm.repeat(3, 1, 1), step)

    if posterior_probs is not None:
        logger.log_image("viz/encoder/latent_posterior", posterior_probs[0].unsqueeze(0), step)

    if dreamed_pixels is not None and obs_pixels is not None:
        actual = (obs_pixels[0] / 255.0).clamp(0, 1)
        frames = dreamed_pixels[:, 0]
        if frames.shape[2:] != actual.shape[1:]:
            frames = F.interpolate(frames, size=actual.shape[1:], mode="bilinear", align_corners=False)
        logger.log_video("viz/wm/dream_video", frames.unsqueeze(0), step, fps=4)

        n_show = min(5, frames.shape[0])
        logger.log_image("viz/wm/dream_strip", torch.cat([frames[i] for i in range(n_show)], dim=2), step)

    if obs_pixels_original is not None:
        logger.log_image("viz/env/frame", (obs_pixels_original[0] / 255.0).clamp(0, 1), step)


def log_progress(
    logger,
    step: int,
    max_steps: int,
    total_wm_loss,
    total_actor_loss,
    total_critic_loss,
    seq_len: int,
    steps_per_sec: float,
    env_steps: int,
    episodes_added: int,
    avg_ep_len: float,
    elapsed_total: float,
):
    """Log training progress to stdout and MLflow (throughput, ETA, env stats)."""
    norm = max(1, seq_len)
    eta_hours = (max_steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0

    print(
        f"Step {step}/{max_steps} | "
        f"{steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h | "
        f"WM: {total_wm_loss.item() / norm:.4f} | "
        f"Actor: {total_actor_loss.item() / norm:.4f} | "
        f"Critic: {total_critic_loss.item() / norm:.4f}"
    )

    m = {
        "train/throughput": steps_per_sec,
        "train/updates_total": float(step),
        "env/frames_total": float(env_steps),
        "train/updates_per_env_step": float(step) / max(1.0, float(env_steps)),
        "env/episodes_total": float(episodes_added),
        "time/elapsed_sec": elapsed_total,
    }
    if avg_ep_len > 0:
        m["env/episode_length"] = avg_ep_len

    logger.log_scalars(m, step)