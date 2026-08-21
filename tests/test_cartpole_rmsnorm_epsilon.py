import pytest
import torch

from scripts.probe_cartpole_rmsnorm_epsilon import (
    REFERENCE_EPSILON,
    rmsnorm_input_effect,
    set_reference_epsilon,
    summarize_comparison,
)


def test_rmsnorm_input_effect_matches_direct_scale_ratio():
    inputs = torch.tensor([[1e-3, -1e-3], [1.0, -1.0]], dtype=torch.float32)
    mean_square, relative = rmsnorm_input_effect(inputs)
    expected = (
        torch.sqrt(
            (mean_square + torch.finfo(torch.float32).eps)
            / (mean_square + REFERENCE_EPSILON)
        )
        - 1.0
    ).abs()
    assert torch.allclose(relative, expected)
    assert relative[0] > 0.8
    assert relative[1] < 1e-3


def test_set_reference_epsilon_preserves_explicit_norms():
    module = torch.nn.Sequential(
        torch.nn.RMSNorm(4), torch.nn.RMSNorm(4, eps=1e-3)
    )
    assert set_reference_epsilon(module) == ["0"]
    assert module[0].eps == pytest.approx(REFERENCE_EPSILON)
    assert module[1].eps == pytest.approx(1e-3)


def test_materiality_gate_requires_local_and_downstream_effects():
    rows = [
        {
            "feature_rms_difference": 0.5,
            "actor_probability_l1": 0.03,
            "critic_value_abs_difference": 0.2,
            "reward_value_abs_difference": 0.0,
            "continuation_probability_abs_difference": 0.0,
            "actor_action_disagreement": 0,
            "prior_category_disagreement": 0.0,
            "posterior_category_disagreement": 0.0,
        }
    ]
    material_norm = {
        "norm": {"relative_output_scale_change": {"p95": 0.02}}
    }
    _downstream, gate = summarize_comparison(rows, material_norm)
    assert gate["local_norm_scale_p95_ge_0.01"] is True
    assert gate["downstream_gate"] is True
    assert gate["materiality_gate_passed"] is True

    immaterial_norm = {
        "norm": {"relative_output_scale_change": {"p95": 0.001}}
    }
    _downstream, gate = summarize_comparison(rows, immaterial_norm)
    assert gate["materiality_gate_passed"] is False
