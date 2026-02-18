"""Checkpoint inspection utility for policy and world model diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.config import Config, atari_pong_config, default_config
from dreamer.envs.utils import create_env
from dreamer.models import (
    initialize_actor,
    initialize_world_model,
    resize_pixels_to_target,
    symlog,
    symexp,
    unimix_logits,
)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_config_from_checkpoint(
    checkpoint_path: Path, config_name: str | None
) -> Config:
    if config_name == "atari_pong":
        return atari_pong_config()
    if config_name == "default":
        return default_config()

    run_dir = checkpoint_path.parent.parent
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text())
        return Config(**data)

    return atari_pong_config()


def load_models(checkpoint_path: Path, cfg: Config, device: str):
    actor = initialize_actor(device, cfg)
    encoder, world_model = initialize_world_model(device, cfg, batch_size=1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"], strict=False)

    actor.eval()
    encoder.eval()
    world_model.eval()

    return actor, encoder, world_model, checkpoint


def decode_twohot_expectation(
    logits: torch.Tensor, b_tensor: torch.Tensor
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.sum(probs * b_tensor, dim=-1)


def run_inspection(
    checkpoint_path: Path,
    cfg: Config,
    device: str,
    episodes: int,
    max_steps_per_episode: int,
    policy_mode: str,
    save_video: bool,
    compose_debug_video: bool,
    video_episodes: int,
    out_dir: Path,
):
    actor, encoder, world_model, ckpt = load_models(checkpoint_path, cfg, device)

    env = create_env(cfg.environment_name, use_pixels=cfg.use_pixels, config=cfg)
    action_meanings = None
    get_action_meanings = getattr(env.unwrapped, "get_action_meanings", None)
    if callable(get_action_meanings):
        try:
            maybe_names = get_action_meanings()
            if isinstance(maybe_names, list) and all(
                isinstance(x, str) for x in maybe_names
            ):
                action_meanings = maybe_names
        except Exception:
            action_meanings = None
    target_size = tuple(cfg.encoder_cnn_target_size) if cfg.use_pixels else None
    n_classes = cfg.d_hidden // 16
    b_tensor = symexp(
        torch.arange(cfg.b_start, cfg.b_end, device=device, dtype=torch.float32)
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "episodes": episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "policy_mode": policy_mode,
    }

    episode_rows = []
    step_rows = []
    action_counter = Counter()

    all_returns = []
    all_lengths = []
    win_count = 0

    reward_pred_abs_errors = []
    continue_pos = []
    continue_neg = []
    recon_l1 = []
    kl_dyn_raw_vals = []
    kl_rep_raw_vals = []
    posterior_entropy_vals = []
    actor_entropy_vals = []

    for ep in range(episodes):
        obs, _ = env.reset()
        h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
        action_onehot = torch.zeros(1, cfg.n_actions, device=device)
        prior_logits_prev = None
        ep_return = 0.0
        ep_steps = 0
        frames = []
        debug_rows = []

        while ep_steps < max_steps_per_episode:
            pixels_model = None
            if cfg.use_pixels:
                pixels = torch.from_numpy(obs["pixels"]).to(device).float()
                pixels = pixels.permute(2, 0, 1).unsqueeze(0)
                if target_size and pixels.shape[-2:] != target_size:
                    pixels_model = resize_pixels_to_target(pixels, target_size)
                else:
                    pixels_model = pixels
                if cfg.n_observations > 0:
                    state_vec = (
                        torch.from_numpy(obs["state"]).to(device).float().unsqueeze(0)
                    )
                    state_vec = symlog(state_vec)
                else:
                    state_vec = torch.zeros(1, 0, device=device)
                encoder_input = {"pixels": pixels_model, "state": state_vec}
            else:
                state_vec = torch.from_numpy(obs).to(device).float().unsqueeze(0)
                state_vec = symlog(state_vec)
                encoder_input = state_vec
                pixels = None

            with torch.no_grad():
                posterior_logits = encoder(encoder_input)
                posterior_mixed = unimix_logits(posterior_logits, unimix_ratio=0.01)
                posterior_probs = F.softmax(posterior_mixed, dim=-1)
                posterior_entropy = (
                    -(posterior_probs * torch.log(posterior_probs + 1e-8))
                    .sum(dim=-1)
                    .mean()
                    .item()
                )
                posterior_entropy_vals.append(posterior_entropy)

                if prior_logits_prev is not None:
                    log_prior = F.log_softmax(prior_logits_prev, dim=-1)
                    log_post = F.log_softmax(posterior_mixed, dim=-1)
                    prior_probs = log_prior.exp()
                    post_probs = log_post.exp()
                    kl_dyn_raw = (
                        (prior_probs * (log_prior - log_post.detach()))
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    kl_rep_raw = (
                        (post_probs * (log_post - log_prior.detach()))
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    kl_dyn_raw_vals.append(float(kl_dyn_raw))
                    kl_rep_raw_vals.append(float(kl_rep_raw))

                z_indices = posterior_mixed.argmax(dim=-1)
                z_sample = F.one_hot(z_indices, num_classes=n_classes).float()
                z_flat = z_sample.view(1, -1)
                z_embed = world_model.z_embedding(z_flat)

                h, prior_logits = world_model.step_dynamics(z_embed, action_onehot, h)
                prior_logits_prev = prior_logits

                obs_recon, reward_logits, continue_logits, actor_input = (
                    world_model.predict_heads(h, z_sample)
                )

                action_logits = actor(actor_input)
                action_probs = F.softmax(action_logits, dim=-1)
                action_entropy = (
                    -(action_probs * torch.log(action_probs + 1e-8))
                    .sum(dim=-1)
                    .mean()
                    .item()
                )
                actor_entropy_vals.append(action_entropy)

                if policy_mode == "sample":
                    action = torch.distributions.Categorical(
                        logits=action_logits
                    ).sample()
                else:
                    action = action_logits.argmax(dim=-1)
                action_idx = int(action.item())
                action_counter[action_idx] += 1
                action_onehot = F.one_hot(action, num_classes=cfg.n_actions).float()

                reward_pred = decode_twohot_expectation(reward_logits, b_tensor).item()
                continue_pred = torch.sigmoid(continue_logits).item()
                kl_dyn_step = kl_dyn_raw_vals[-1] if kl_dyn_raw_vals else 0.0
                kl_rep_step = kl_rep_raw_vals[-1] if kl_rep_raw_vals else 0.0

                pixel_recon = obs_recon.get("pixels")
                if (
                    cfg.use_pixels
                    and pixels is not None
                    and pixels_model is not None
                    and isinstance(pixel_recon, torch.Tensor)
                ):
                    recon = torch.sigmoid(pixel_recon).clamp(0, 1)
                    target = (pixels_model / 255.0).clamp(0, 1)
                    recon_l1.append(torch.mean(torch.abs(recon - target)).item())
                    recon_img = (
                        (recon[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    err_map = (
                        (
                            torch.mean(torch.abs(recon - target), dim=1, keepdim=True)[
                                0
                            ]
                            .repeat(3, 1, 1)
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                            * 255.0
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                else:
                    recon_img = None
                    err_map = None

            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            if save_video and ep < video_episodes and cfg.use_pixels:
                frames.append(obs["pixels"])
                debug_rows.append(
                    {
                        "action": action_idx,
                        "action_probs": action_probs[0].detach().cpu().numpy().tolist(),
                        "reward": float(reward),
                        "reward_pred": float(reward_pred),
                        "continue_pred": float(continue_pred),
                        "kl_dyn": float(kl_dyn_step),
                        "kl_rep": float(kl_rep_step),
                        "actor_entropy": float(action_entropy),
                        "posterior_entropy": float(posterior_entropy),
                        "recon_img": recon_img,
                        "err_map": err_map,
                    }
                )

            reward_pred_abs_errors.append(abs(float(reward) - float(reward_pred)))
            done = bool(terminated or truncated)
            if done:
                continue_pos.append(continue_pred)
            else:
                continue_neg.append(continue_pred)

            ep_return += float(reward)
            ep_steps += 1

            step_rows.append(
                {
                    "episode": ep,
                    "t": ep_steps,
                    "action": action_idx,
                    "reward": float(reward),
                    "reward_pred": float(reward_pred),
                    "continue_pred": float(continue_pred),
                    "posterior_entropy": float(posterior_entropy),
                    "actor_entropy": float(action_entropy),
                }
            )

            obs = next_obs
            if done:
                break

        if save_video and ep < video_episodes and frames:
            video_path = out_dir / f"episode_{ep:02d}_raw.mp4"
            imageio.mimwrite(video_path, frames, fps=20)
            if compose_debug_video:
                debug_path = out_dir / f"episode_{ep:02d}_debug.mp4"
                render_debug_video(
                    frames,
                    debug_rows,
                    cfg.n_actions,
                    debug_path,
                    action_meanings=action_meanings,
                )

        all_returns.append(ep_return)
        all_lengths.append(ep_steps)
        if ep_return > 0:
            win_count += 1
        episode_rows.append(
            {
                "episode": ep,
                "return": ep_return,
                "length": ep_steps,
                "won": int(ep_return > 0),
            }
        )

    env.close()

    total_actions = sum(action_counter.values())
    action_hist = {
        str(i): (action_counter[i] / total_actions if total_actions > 0 else 0.0)
        for i in range(cfg.n_actions)
    }

    summary.update(
        {
            "avg_return": float(sum(all_returns) / max(1, len(all_returns))),
            "avg_length": float(sum(all_lengths) / max(1, len(all_lengths))),
            "win_rate": float(win_count / max(1, len(all_returns))),
            "action_hist_fraction": action_hist,
            "wm_reward_pred_abs_error_mean": float(
                sum(reward_pred_abs_errors) / max(1, len(reward_pred_abs_errors))
            ),
            "wm_continue_pred_done_mean": float(
                sum(continue_pos) / max(1, len(continue_pos))
            ),
            "wm_continue_pred_not_done_mean": float(
                sum(continue_neg) / max(1, len(continue_neg))
            ),
            "wm_recon_l1_mean": float(sum(recon_l1) / max(1, len(recon_l1))),
            "kl_dynamics_raw_mean": float(
                sum(kl_dyn_raw_vals) / max(1, len(kl_dyn_raw_vals))
            ),
            "kl_representation_raw_mean": float(
                sum(kl_rep_raw_vals) / max(1, len(kl_rep_raw_vals))
            ),
            "posterior_entropy_mean": float(
                sum(posterior_entropy_vals) / max(1, len(posterior_entropy_vals))
            ),
            "actor_entropy_mean": float(
                sum(actor_entropy_vals) / max(1, len(actor_entropy_vals))
            ),
        }
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length", "won"])
        writer.writeheader()
        writer.writerows(episode_rows)

    if step_rows:
        with (out_dir / "steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "t",
                    "action",
                    "reward",
                    "reward_pred",
                    "continue_pred",
                    "posterior_entropy",
                    "actor_entropy",
                ],
            )
            writer.writeheader()
            writer.writerows(step_rows)

    return summary


def _sparkline(
    values: list[float], w: int, h: int, color: tuple[int, int, int]
) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if len(values) < 2:
        return canvas
    arr = np.array(values, dtype=np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0
    xs = np.linspace(0, w - 1, num=len(arr)).astype(np.int32)
    ys = (h - 1 - (arr - vmin) / (vmax - vmin) * (h - 1)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, 2)
    return canvas


def render_debug_video(
    frames: list[np.ndarray],
    debug_rows: list[dict],
    n_actions: int,
    out_path: Path,
    action_meanings: list[str] | None = None,
):
    width, height = 1280, 720
    left_w = 720
    right_w = width - left_w
    reward_hist: list[float] = []
    reward_pred_hist: list[float] = []
    kl_dyn_hist: list[float] = []
    kl_rep_hist: list[float] = []

    writer = imageio.get_writer(out_path, fps=20)
    for i, frame in enumerate(frames):
        dbg = debug_rows[i] if i < len(debug_rows) else None
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Left panel: environment frame
        rgb = frame
        left_h = int(left_w * rgb.shape[0] / max(1, rgb.shape[1]))
        left_h = min(left_h, height)
        resized = cv2.resize(rgb, (left_w, left_h), interpolation=cv2.INTER_AREA)
        canvas[:left_h, :left_w, :] = resized

        # Right panel background
        panel = np.zeros((height, right_w, 3), dtype=np.uint8)
        panel[:] = (16, 16, 16)

        y = 30
        if dbg is not None:
            reward_hist.append(dbg["reward"])
            reward_pred_hist.append(dbg["reward_pred"])
            kl_dyn_hist.append(dbg["kl_dyn"])
            kl_rep_hist.append(dbg["kl_rep"])

            cv2.putText(
                panel,
                f"t={i + 1}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (240, 240, 240),
                2,
            )
            y += 34
            cv2.putText(
                panel,
                f"a={dbg['action']} ({action_meanings[dbg['action']] if action_meanings and dbg['action'] < len(action_meanings) else 'UNK'})  r={dbg['reward']:+.2f}  r_pred={dbg['reward_pred']:+.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            y += 28
            cv2.putText(
                panel,
                f"continue={dbg['continue_pred']:.3f}  actor_H={dbg['actor_entropy']:.3f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            y += 28
            cv2.putText(
                panel,
                f"kl_dyn={dbg['kl_dyn']:.3f}  kl_rep={dbg['kl_rep']:.3f}  post_H={dbg['posterior_entropy']:.3f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            y += 22

            # Action probabilities bar chart
            probs = np.array(dbg["action_probs"], dtype=np.float32)
            probs = probs[:n_actions]
            bar_top = y + 10
            bar_h = 120
            bar_w = right_w - 40
            cv2.rectangle(
                panel, (20, bar_top), (20 + bar_w, bar_top + bar_h), (60, 60, 60), 1
            )
            for a, p in enumerate(probs):
                x0 = 24 + int(a * (bar_w - 8) / max(1, n_actions))
                x1 = 24 + int((a + 1) * (bar_w - 8) / max(1, n_actions)) - 4
                bh = int(p * (bar_h - 8))
                cv2.rectangle(
                    panel,
                    (x0, bar_top + bar_h - 4 - bh),
                    (x1, bar_top + bar_h - 4),
                    (100, 200, 255),
                    -1,
                )
                cv2.putText(
                    panel,
                    str(a),
                    (x0, bar_top + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (180, 180, 180),
                    1,
                )
            y = bar_top + bar_h + 28

            # Sparklines
            rw = _sparkline(reward_hist[-200:], right_w - 40, 70, (80, 220, 120))
            rp = _sparkline(reward_pred_hist[-200:], right_w - 40, 70, (220, 180, 80))
            kd = _sparkline(kl_dyn_hist[-200:], right_w - 40, 70, (120, 180, 255))
            kr = _sparkline(kl_rep_hist[-200:], right_w - 40, 70, (255, 140, 120))
            panel[y : y + 70, 20 : 20 + right_w - 40] = rw
            cv2.putText(
                panel,
                "reward",
                (24, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (210, 210, 210),
                1,
            )
            y += 80
            panel[y : y + 70, 20 : 20 + right_w - 40] = rp
            cv2.putText(
                panel,
                "reward_pred",
                (24, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (210, 210, 210),
                1,
            )
            y += 80
            panel[y : y + 70, 20 : 20 + right_w - 40] = kd
            cv2.putText(
                panel,
                "kl_dyn",
                (24, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (210, 210, 210),
                1,
            )
            y += 80
            panel[y : y + 70, 20 : 20 + right_w - 40] = kr
            cv2.putText(
                panel,
                "kl_rep",
                (24, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (210, 210, 210),
                1,
            )
            y += 86

            # Recon and error thumbnails (fixed near bottom to avoid overflow)
            y_thumb = max(0, height - 140)
            if isinstance(dbg.get("recon_img"), np.ndarray):
                thumb_w = (right_w - 60) // 2
                thumb_h = 120
                recon = cv2.resize(
                    dbg["recon_img"], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA
                )
                panel[y_thumb : y_thumb + thumb_h, 20 : 20 + thumb_w] = recon
                cv2.putText(
                    panel,
                    "recon",
                    (24, y_thumb + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                )
            if isinstance(dbg.get("err_map"), np.ndarray):
                thumb_w = (right_w - 60) // 2
                thumb_h = 120
                err = cv2.resize(
                    dbg["err_map"], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA
                )
                panel[y_thumb : y_thumb + thumb_h, 40 + thumb_w : 40 + 2 * thumb_w] = (
                    err
                )
                cv2.putText(
                    panel,
                    "error",
                    (44 + thumb_w, y_thumb + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                )

        canvas[:, left_w:, :] = panel
        writer.append_data(canvas)
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Dreamer checkpoint behavior and WM diagnostics"
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint_step_*.pt")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=2000,
        help="Per-episode hard step cap",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Optional config override when config.json is unavailable",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="inspection",
        help="Root directory for inspector outputs",
    )
    parser.add_argument(
        "--use_timestamp",
        action="store_true",
        help="Append timestamp directory under output path",
    )
    parser.add_argument("--save_video", action="store_true", help="Save rollout videos")
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="argmax",
        choices=["argmax", "sample"],
        help="Action selection for inspection episodes",
    )
    parser.add_argument(
        "--compose_debug_video",
        action="store_true",
        help="Compose side-by-side debug video with diagnostics panel",
    )
    parser.add_argument(
        "--video_episodes", type=int, default=1, help="How many episodes to save as mp4"
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    cfg = infer_config_from_checkpoint(checkpoint_path, args.config_name)

    run_name = checkpoint_path.parent.parent.name
    m = re.search(r"checkpoint_step_(\d+)$", checkpoint_path.stem)
    step_str = m.group(1) if m else checkpoint_path.stem
    out_dir = Path(args.output_root) / run_name / f"step_{step_str}" / args.policy_mode
    if args.use_timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir / stamp

    if out_dir.exists() and not args.use_timestamp:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_inspection(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        device=device,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        policy_mode=args.policy_mode,
        save_video=args.save_video,
        compose_debug_video=args.compose_debug_video,
        video_episodes=args.video_episodes,
        out_dir=out_dir,
    )

    print("Inspection complete")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output: {out_dir}")
    print(f"  avg_return: {summary['avg_return']:.3f}")
    print(f"  win_rate: {summary['win_rate']:.3f}")
    print(
        f"  wm_reward_pred_abs_error_mean: {summary['wm_reward_pred_abs_error_mean']:.3f}"
    )
    print(f"  wm_recon_l1_mean: {summary['wm_recon_l1_mean']:.4f}")


if __name__ == "__main__":
    main()
