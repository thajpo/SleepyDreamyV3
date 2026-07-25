import pytest

from scripts.probe_cartpole_carry_parity import summarize


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
