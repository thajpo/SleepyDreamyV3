"""Checkpoint visualization utility.

This script provides a stable visualization entrypoint by reusing the
inspection rollout pipeline and writing episode MP4 files.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from dreamer.inspect import infer_config_from_checkpoint, resolve_device, run_inspection


def main():
    parser = argparse.ArgumentParser(
        description="Visualize checkpoint behavior as rollout videos"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Max environment steps per rollout episode",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of rollout episodes"
    )
    parser.add_argument(
        "--policy_mode",
        choices=["argmax", "sample"],
        default="argmax",
        help="Action selection mode",
    )
    parser.add_argument(
        "--use_actor",
        action="store_true",
        help="Kept for backward compatibility; policy still comes from checkpoint actor",
    )
    parser.add_argument(
        "--output",
        default="videos",
        help="Root output directory for generated videos",
    )
    parser.add_argument(
        "--compose_debug_video",
        action="store_true",
        help="Also save side-by-side debug overlays",
    )
    parser.add_argument(
        "--dream_snippets_per_episode",
        type=int,
        default=3,
        help="How many dreamed rollout snippets to export per saved episode",
    )
    parser.add_argument(
        "--dream_snippet_horizon",
        type=int,
        default=30,
        help="Imagined horizon length (frames) for each dream snippet",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto/cuda/mps/cpu",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Optional config preset override when run config.json is unavailable",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    cfg = infer_config_from_checkpoint(checkpoint_path, args.config_name)

    run_name = checkpoint_path.parent.parent.name
    match = re.search(r"checkpoint_step_(\d+)$", checkpoint_path.stem)
    step_str = match.group(1) if match else checkpoint_path.stem
    out_dir = Path(args.output) / run_name / f"step_{step_str}" / args.policy_mode

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_inspection(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        device=device,
        episodes=max(1, int(args.num_rollouts)),
        max_steps_per_episode=max(1, int(args.steps)),
        policy_mode=args.policy_mode,
        save_video=True,
        compose_debug_video=bool(args.compose_debug_video),
        video_episodes=max(1, int(args.num_rollouts)),
        out_dir=out_dir,
        dream_snippets_per_episode=max(0, int(args.dream_snippets_per_episode)),
        dream_snippet_horizon=max(1, int(args.dream_snippet_horizon)),
    )

    print("Visualization complete")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output: {out_dir}")
    print(f"  avg_return: {summary['avg_return']:.3f}")
    print(f"  win_rate: {summary['win_rate']:.3f}")


if __name__ == "__main__":
    main()
