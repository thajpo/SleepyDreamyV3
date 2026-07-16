import torch

from dreamer.runtime.model_transport import (
    deserialize_model_states,
    serialize_model_states,
)


def test_model_update_is_self_contained_bytes():
    original = {
        "actor": {"weight": torch.arange(4, dtype=torch.float32)},
        "encoder": {},
        "world_model": {},
    }

    payload = serialize_model_states(original)
    original["actor"]["weight"].fill_(-1)
    restored = deserialize_model_states(payload, "cpu")

    assert isinstance(payload, bytes)
    assert torch.equal(
        restored["actor"]["weight"],
        torch.arange(4, dtype=torch.float32),
    )
