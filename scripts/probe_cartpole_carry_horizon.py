#!/usr/bin/env python3
"""Compare continuous and truncated RSSM carry on fixed CartPole histories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dreamer.inspect import resolve_device
from dreamer.models import symlog, symexp, symexp_twohot_bins, twohot_expectation

if __package__:
    from scripts.probe_cartpole_q import load_checkpoint_models
else:
    from probe_cartpole_q import load_checkpoint_models  # type: ignore[import-not-found]


def _mean(rows: list[dict], key: str) -> float | None:
    return float(np.mean([float(row[key]) for row in rows])) if rows else None


def _time_bin(timestep: int) -> str:
    if timestep < 16:
        return "0-15"
    if timestep < 32:
        return "16-31"
    if timestep < 64:
        return "32-63"
    if timestep < 128:
        return "64-127"
    return "128+"


def _position_bin(x: float) -> str:
    value = abs(x)
    if value < 0.5:
        return "0.0-0.5"
    if value < 1.0:
        return "0.5-1.0"
    if value < 1.5:
        return "1.0-1.5"
    if value < 2.0:
        return "1.5-2.0"
    return "2.0+"


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"states": 0}
    return {
        "states": len(rows),
        "actor_action_disagreement": float(
            np.mean(
                [
                    int(row["full_actor_action"])
                    != int(row["local_actor_action"])
                    for row in rows
                ]
            )
        ),
        "mean_abs_actor_logit_delta_difference": _mean(
            rows, "abs_actor_logit_delta_difference"
        ),
        "mean_feature_rms_distance": _mean(rows, "feature_rms_distance"),
        "mean_abs_critic_value_difference": _mean(
            rows, "abs_critic_value_difference"
        ),
        "mean_full_decoder_mse": _mean(rows, "full_decoder_mse"),
        "mean_local_decoder_mse": _mean(rows, "local_decoder_mse"),
        "mean_full_decoder_x_mse": _mean(rows, "full_decoder_x_mse"),
        "mean_local_decoder_x_mse": _mean(rows, "local_decoder_x_mse"),
    }


def summarize_context_rows(rows: list[dict]) -> dict:
    """Summarize carry sensitivity by checkpoint, context, time, and position."""
    if not rows:
        raise ValueError("carry-horizon probe produced no rows")
    summary: dict[str, object] = {"rows": len(rows), "checkpoints": {}}
    checkpoints: dict[str, dict] = summary["checkpoints"]  # type: ignore[assignment]
    checkpoint_names = sorted({str(row["checkpoint"]) for row in rows})
    for checkpoint in checkpoint_names:
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        contexts = {}
        for context in sorted({int(row["context_rows"]) for row in checkpoint_rows}):
            selected = [
                row for row in checkpoint_rows if int(row["context_rows"]) == context
            ]
            contexts[str(context)] = {
                **_aggregate(selected),
                "by_time": {
                    name: _aggregate(
                        [row for row in selected if row["time_bin"] == name]
                    )
                    for name in ("0-15", "16-31", "32-63", "64-127", "128+")
                    if any(row["time_bin"] == name for row in selected)
                },
                "by_abs_x": {
                    name: _aggregate(
                        [row for row in selected if row["abs_x_bin"] == name]
                    )
                    for name in (
                        "0.0-0.5",
                        "0.5-1.0",
                        "1.0-1.5",
                        "1.5-2.0",
                        "2.0+",
                    )
                    if any(row["abs_x_bin"] == name for row in selected)
                },
            }
        checkpoints[checkpoint] = {"contexts": contexts}
    return summary


def _initial_carry(cfg, world_model, device: str):
    h = torch.zeros(1, cfg.d_hidden * cfg.rnn_n_blocks, device=device)
    z = torch.zeros(
        1, world_model.n_latents, world_model.n_classes, device=device
    )
    z_embed = world_model.z_embedding(z.view(1, -1))
    return h, z_embed


def _observe(encoder, world_model, h, z_embed, previous_action, state):
    h, _ = world_model.step_dynamics(z_embed, previous_action, h)
    state_tensor = torch.from_numpy(state).to(h.device).float().unsqueeze(0)
    tokens = encoder(symlog(state_tensor))
    posterior_logits = world_model.compute_posterior(h, tokens)
    posterior = F.one_hot(
        posterior_logits.argmax(dim=-1), num_classes=world_model.n_classes
    ).float()
    feature = world_model.join_h_and_z(h, posterior)
    z_embed = world_model.z_embedding(posterior.view(1, -1))
    return h, z_embed, feature


def _feature_readout(actor, critic, world_model, feature, state, bins):
    actor_logits = actor(feature)
    actor_delta = float((actor_logits[0, 1] - actor_logits[0, 0]).item())
    actor_action = int(actor_logits.argmax(dim=-1).item())
    critic_value = float(twohot_expectation(critic(feature), bins).item())
    reconstruction = (
        symexp(world_model.decoder(feature)["state"])
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    error = reconstruction - state
    return {
        "actor_delta": actor_delta,
        "actor_action": actor_action,
        "critic_value": critic_value,
        "decoder_mse": float(np.mean(error**2)),
        "decoder_x_mse": float(error[0] ** 2),
    }


def _load_histories(path: Path) -> dict[int, list[dict]]:
    histories: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            histories[int(row["episode"])].append(
                {
                    "episode": int(row["episode"]),
                    "t": int(row["t"]),
                    "state": np.asarray(
                        [row["x"], row["x_dot"], row["theta"], row["theta_dot"]],
                        dtype=np.float32,
                    ),
                    "source_action": int(row["source_action"]),
                }
            )
    for episode_rows in histories.values():
        episode_rows.sort(key=lambda row: int(row["t"]))
    return dict(histories)


@torch.no_grad()
def probe_checkpoint(
    checkpoint_path: Path,
    histories: dict[int, list[dict]],
    *,
    contexts: list[int],
    device: str,
) -> list[dict]:
    (
        cfg,
        actor,
        critic,
        _q_critic,
        encoder,
        world_model,
        checkpoint,
        _critic_key,
        _q_key,
    ) = load_checkpoint_models(checkpoint_path, device, critic_source="online")
    bins = symexp_twohot_bins(
        cfg.b_start,
        cfg.b_end,
        int(getattr(cfg, "num_bins", 255)),
        device=device,
        dtype=torch.float32,
    )
    step = int(checkpoint.get("step", checkpoint.get("train_step", -1)))
    checkpoint_name = f"step_{step}"
    output: list[dict] = []
    h_backup = world_model.h_prev.clone()
    try:
        for episode, history in sorted(histories.items()):
            full_h, full_z = _initial_carry(cfg, world_model, device)
            previous_action = torch.zeros(1, cfg.n_actions, device=device)
            full_features = []
            full_readouts = []
            for row in history:
                full_h, full_z, feature = _observe(
                    encoder,
                    world_model,
                    full_h,
                    full_z,
                    previous_action,
                    row["state"],
                )
                full_features.append(feature.clone())
                full_readouts.append(
                    _feature_readout(
                        actor, critic, world_model, feature, row["state"], bins
                    )
                )
                previous_action = F.one_hot(
                    torch.tensor([row["source_action"]], device=device),
                    num_classes=cfg.n_actions,
                ).float()

            for index, row in enumerate(history):
                for context in contexts:
                    start = max(0, index - context + 1)
                    local_h, local_z = _initial_carry(cfg, world_model, device)
                    if start == 0:
                        local_previous_action = torch.zeros(
                            1, cfg.n_actions, device=device
                        )
                    else:
                        local_previous_action = F.one_hot(
                            torch.tensor(
                                [history[start - 1]["source_action"]], device=device
                            ),
                            num_classes=cfg.n_actions,
                        ).float()
                    local_feature = None
                    for local_index in range(start, index + 1):
                        local_row = history[local_index]
                        local_h, local_z, local_feature = _observe(
                            encoder,
                            world_model,
                            local_h,
                            local_z,
                            local_previous_action,
                            local_row["state"],
                        )
                        local_previous_action = F.one_hot(
                            torch.tensor(
                                [local_row["source_action"]], device=device
                            ),
                            num_classes=cfg.n_actions,
                        ).float()
                    assert local_feature is not None
                    local_readout = _feature_readout(
                        actor,
                        critic,
                        world_model,
                        local_feature,
                        row["state"],
                        bins,
                    )
                    full_feature = full_features[index]
                    full_readout = full_readouts[index]
                    output.append(
                        {
                            "checkpoint": checkpoint_name,
                            "episode": episode,
                            "t": row["t"],
                            "context_rows": context,
                            "time_bin": _time_bin(int(row["t"])),
                            "abs_x_bin": _position_bin(float(row["state"][0])),
                            "full_actor_action": full_readout["actor_action"],
                            "local_actor_action": local_readout["actor_action"],
                            "full_actor_logit_delta": full_readout["actor_delta"],
                            "local_actor_logit_delta": local_readout["actor_delta"],
                            "abs_actor_logit_delta_difference": abs(
                                float(full_readout["actor_delta"])
                                - float(local_readout["actor_delta"])
                            ),
                            "feature_rms_distance": float(
                                torch.sqrt(
                                    torch.mean((full_feature - local_feature) ** 2)
                                ).item()
                            ),
                            "full_critic_value": full_readout["critic_value"],
                            "local_critic_value": local_readout["critic_value"],
                            "abs_critic_value_difference": abs(
                                float(full_readout["critic_value"])
                                - float(local_readout["critic_value"])
                            ),
                            "full_decoder_mse": full_readout["decoder_mse"],
                            "local_decoder_mse": local_readout["decoder_mse"],
                            "full_decoder_x_mse": full_readout["decoder_x_mse"],
                            "local_decoder_x_mse": local_readout["decoder_x_mse"],
                        }
                    )
    finally:
        world_model.h_prev = h_backup
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contexts", nargs="+", type=int, default=[4, 8, 16, 32])
    args = parser.parse_args()
    if any(context <= 0 for context in args.contexts):
        parser.error("--contexts must contain positive integers")
    device = resolve_device(args.device)
    histories = _load_histories(args.rows)
    rows = []
    for checkpoint in args.checkpoints:
        print(f"Probing carry horizon for {checkpoint} on {device}...")
        rows.extend(
            probe_checkpoint(
                checkpoint.resolve(),
                histories,
                contexts=sorted(set(args.contexts)),
                device=device,
            )
        )
    summary = summarize_context_rows(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
