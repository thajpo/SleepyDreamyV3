import pytest
import torch

from scripts.probe_slow_value_regularizer import slow_regularizer_metrics


def test_slow_regularizer_probe_distinguishes_distribution_from_mean_target():
    bins = torch.tensor([-1.0, 0.0, 1.0])
    online_logits = torch.tensor([[0.0, 3.0, 0.0]])
    # Symmetric endpoint mass has decoded mean zero but is not the zero two-hot.
    slow_logits = torch.tensor([[3.0, 0.0, 3.0]])

    metrics = slow_regularizer_metrics(online_logits, slow_logits, bins)

    assert metrics["states"] == 1
    assert metrics["mean_target_total_variation"] > 0.8
    assert metrics["mean_slow_target_entropy"] > 0.7
    assert metrics["mean_reference_target_entropy"] == pytest.approx(0.0)
    assert metrics["mean_gradient_cosine"] < 0.0


def test_slow_regularizer_probe_matches_identical_point_targets():
    bins = torch.tensor([-1.0, 0.0, 1.0])
    online_logits = torch.tensor([[0.0, 1.0, 0.0]])
    slow_logits = torch.tensor([[-100.0, 100.0, -100.0]])

    metrics = slow_regularizer_metrics(online_logits, slow_logits, bins)

    assert metrics["mean_target_total_variation"] == pytest.approx(0.0)
    assert metrics["mean_gradient_cosine"] == pytest.approx(1.0)
