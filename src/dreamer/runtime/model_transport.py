"""Process-safe serialization for trainer-to-collector model updates."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

import torch


def serialize_model_states(states: Mapping[str, Any]) -> bytes:
    """Copy model state into queue-owned bytes, avoiding shared-memory handles."""
    buffer = BytesIO()
    torch.save(dict(states), buffer)
    return buffer.getvalue()


def deserialize_model_states(payload: bytes, device: str) -> dict[str, Any]:
    """Restore a self-contained model update in the receiving process."""
    states = torch.load(
        BytesIO(payload),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(states, dict):
        raise TypeError("model update payload must contain a state dictionary")
    return states
