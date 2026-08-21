"""Atomic, state-only replay evidence artifacts for research diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np


REQUIRED_ARRAYS = frozenset(
    {
        "states",
        "actions",
        "rewards",
        "is_last",
        "is_terminal",
        "mask",
        "is_first",
        "start_collector_id",
        "start_episode_id",
        "start_offset",
        "sample_count",
        "sequence_length",
        "seed",
        "selector",
        "train_step",
        "checkpoint_name",
    }
)


def save_replay_evidence(
    destination: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    train_step: int,
    checkpoint_name: str,
) -> str:
    """Write a replay evidence sample atomically without pickle objects."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **arrays,
        "train_step": np.asarray(train_step, dtype=np.int64),
        "checkpoint_name": np.asarray(checkpoint_name),
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def load_replay_evidence(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a replay evidence artifact without enabling pickle."""
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"replay evidence is missing required arrays: {names}")
        result = {name: archive[name] for name in archive.files}

    count = int(result["sample_count"])
    length = int(result["sequence_length"])
    sequence_fields = (
        "states",
        "actions",
        "rewards",
        "is_last",
        "is_terminal",
        "mask",
        "is_first",
    )
    for name in sequence_fields:
        if result[name].shape[:2] != (count, length):
            raise ValueError(
                f"replay evidence {name} shape {result[name].shape} does not "
                f"match ({count}, {length}, ...)"
            )
    for name in ("start_collector_id", "start_episode_id", "start_offset"):
        if result[name].shape != (count,):
            raise ValueError(
                f"replay evidence {name} shape {result[name].shape} does not "
                f"match ({count},)"
            )
    if result["states"].ndim != 3 or result["states"].shape[-1] == 0:
        raise ValueError("replay evidence must contain vector observations")
    if str(result["selector"]) != "uniform_stream":
        raise ValueError("unsupported replay evidence selector")
    return result
