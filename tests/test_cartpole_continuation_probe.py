import pytest

from scripts.probe_cartpole_continuation import (
    binary_roc_auc,
    continuation_error_components,
    summarize_channel,
)


def test_binary_roc_auc_handles_order_and_ties():
    assert binary_roc_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert binary_roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert binary_roc_auc([0.5, 0.5], [True, False]) == 0.5
    assert binary_roc_auc([0.5], [True]) is None


def test_continuation_error_decomposition_is_exact():
    components = continuation_error_components(
        prior_discount=0.8,
        posterior_discount=0.3,
        target_discount=0.0,
    )

    assert components["prior_error"] == pytest.approx(
        components["posterior_error"] + components["transport_error"]
    )


def test_channel_summary_marks_missing_terminal_class_unavailable():
    rows = [
        {"terminal": False, "target_discount": 0.997, "prediction": 0.99},
        {"terminal": False, "target_discount": 0.997, "prediction": 0.98},
    ]

    summary = summarize_channel(rows, "prediction", gamma=0.997)

    assert summary["terminal_mean"] is None
    assert summary["failure_roc_auc"] is None
    assert summary["balanced_accuracy_at_half_discount"] is None
