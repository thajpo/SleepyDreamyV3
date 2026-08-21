from dataclasses import asdict

import pytest
import torch

from dreamer.models import initialize_actor, initialize_world_model
from scripts.probe_cartpole_carry_parity import (
    load_probe_models,
    reference_config,
    summarize,
)


def test_carry_parity_summary_preserves_gate_statistics() -> None:
    rows = [
        {
            "feature_relative_l2": 0.01,
            "feature_cosine": 0.99,
            "h_relative_l2": 0.02,
            "z_mean_absolute_error": 0.03,
            "actor_probability_l1": 0.04,
            "actor_action_agreement": 1.0,
        },
        {
            "feature_relative_l2": 0.03,
            "feature_cosine": 1.0,
            "h_relative_l2": 0.04,
            "z_mean_absolute_error": 0.05,
            "actor_probability_l1": 0.06,
            "actor_action_agreement": 0.0,
        },
    ]

    summary = summarize(rows)

    assert summary["comparisons"] == 2
    assert summary["feature_relative_l2_median"] == pytest.approx(0.02)
    assert summary["feature_cosine_min"] == pytest.approx(0.99)
    assert summary["actor_action_agreement_mean"] == pytest.approx(0.5)


def test_carry_parity_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize([])


def test_carry_parity_loads_trained_checkpoint(tmp_path) -> None:
    config = reference_config(seed=3)
    encoder, world_model = initialize_world_model("cpu", config, batch_size=1)
    actor = initialize_actor("cpu", config)
    checkpoint_path = tmp_path / "checkpoint_step_17.pt"
    torch.save(
        {
            "step": 17,
            "config_snapshot": asdict(config),
            "encoder": encoder.state_dict(),
            "world_model": world_model.state_dict(),
            "actor": actor.state_dict(),
        },
        checkpoint_path,
    )

    loaded_config, loaded_encoder, loaded_world_model, loaded_actor, step = (
        load_probe_models(seed=999, checkpoint_path=checkpoint_path)
    )

    assert loaded_config.architecture_contract == "reference_v3_state"
    assert step == 17
    assert loaded_encoder.training is False
    assert loaded_world_model.training is False
    assert loaded_actor.training is False
