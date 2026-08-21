import pytest

from scripts.probe_cartpole_recovery_value_representability import (
    paired_action_metrics,
    q_from_continuation,
)


def test_q_from_continuation_aligns_reward_with_successor_value():
    assert q_from_continuation(1.0, 0.9, 4.0) == pytest.approx(4.6)
    assert q_from_continuation(1.0, 0.9, 0.0) == pytest.approx(1.0)


def test_paired_action_metrics_reports_balanced_ordering_and_margin_correlation():
    rows = [
        {"base_index": 0, "action": 0, "target_q": 1.0, "predicted_q": 1.1},
        {"base_index": 0, "action": 1, "target_q": 3.0, "predicted_q": 2.9},
        {"base_index": 1, "action": 0, "target_q": 4.0, "predicted_q": 3.8},
        {"base_index": 1, "action": 1, "target_q": 1.0, "predicted_q": 1.2},
    ]

    summary = paired_action_metrics(rows, "predicted_q", policy="target")

    assert summary["actionable_pairs"] == 2
    assert summary["real_preference_hist"] == {"1": 1, "0": 1}
    assert summary["predicted_preference_hist"] == {"1": 1, "0": 1}
    assert summary["balanced_accuracy"] == 1.0
    assert summary["margin_pearson"] == pytest.approx(1.0)


def test_paired_action_metrics_excludes_ties_and_handles_one_class():
    rows = [
        {"base_index": 0, "action": 0, "heuristic_q": 2.0, "predicted_q": 1.0},
        {"base_index": 0, "action": 1, "heuristic_q": 2.0, "predicted_q": 3.0},
        {"base_index": 1, "action": 0, "heuristic_q": 1.0, "predicted_q": 1.0},
        {"base_index": 1, "action": 1, "heuristic_q": 4.0, "predicted_q": 2.0},
    ]

    summary = paired_action_metrics(rows, "predicted_q", policy="heuristic")

    assert summary["actionable_pairs"] == 1
    assert summary["real_preference_hist"] == {"1": 1}
    assert summary["balanced_accuracy"] == 1.0
    assert summary["margin_pearson"] is None
