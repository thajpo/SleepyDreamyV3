import pytest

from scripts.probe_cartpole_continuation import (
    binary_roc_auc,
    continuation_error_components,
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
