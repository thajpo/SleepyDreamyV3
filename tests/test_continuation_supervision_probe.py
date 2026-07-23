import pytest
import torch

from scripts.probe_cartpole_continuation_supervision import (
    build_failure_head,
    summarize_failure_scores,
)


def test_failure_score_summary_reports_auc_and_balanced_accuracy():
    summary = summarize_failure_scores(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([True, True, False, False]),
    )

    assert summary["failure_roc_auc"] == pytest.approx(1.0)
    assert summary["balanced_accuracy_at_half"] == pytest.approx(1.0)
    assert summary["terminal_recall_at_half"] == pytest.approx(1.0)
    assert summary["live_recall_at_half"] == pytest.approx(1.0)


def test_failure_head_matches_authored_continuation_architecture():
    head = build_failure_head(7, 11)
    assert head(torch.zeros(3, 7)).shape == (3, 1)
    assert isinstance(head[1], torch.nn.RMSNorm)
    assert isinstance(head[2], torch.nn.SiLU)
