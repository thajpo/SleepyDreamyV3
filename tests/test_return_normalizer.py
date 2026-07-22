import pytest
import torch

from dreamer.models import compute_reinforce_actor_loss
from dreamer.trainer.forward import calculate_return_normalizer_update


def test_return_normalizer_first_update_is_not_debiased() -> None:
    returns = torch.arange(100, dtype=torch.float32).view(10, 10) * 100
    update = calculate_return_normalizer_update(
        [(returns, torch.ones(10))],
        0.0,
        0.0,
        rate=0.01,
    )

    assert update is not None
    scale, lo, hi = update
    assert lo == pytest.approx(4.95)
    assert hi == pytest.approx(94.05)
    assert scale == pytest.approx(89.10)


def test_return_normalizer_aggregates_all_valid_imagination_starts() -> None:
    first = torch.tensor([[0.0, 1_000.0], [10.0, 1_000.0]])
    second = torch.tensor([[100.0, 200.0], [110.0, 210.0]])
    batches = [
        (first, torch.tensor([1.0, 0.0])),
        (second, torch.tensor([1.0, 1.0])),
    ]

    update = calculate_return_normalizer_update(
        batches,
        0.0,
        0.0,
        rate=1.0,
    )

    assert update is not None
    scale, lo, hi = update
    expected = torch.tensor([0.0, 10.0, 100.0, 110.0, 200.0, 210.0])
    assert lo == pytest.approx(torch.quantile(expected, 0.05).item())
    assert hi == pytest.approx(torch.quantile(expected, 0.95).item())
    assert scale == pytest.approx(hi - lo)


def test_current_return_scale_controls_unzscored_actor_loss() -> None:
    lambda_returns = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    mask = torch.ones(2)
    update = calculate_return_normalizer_update(
        [(lambda_returns, mask)],
        0.0,
        0.0,
        rate=1.0,
    )
    assert update is not None
    scale, _lo, _hi = update

    values = torch.zeros(3, 2)
    continues = torch.full((2, 2), 20.0)
    logits = torch.zeros(2, 2, 2)
    actions = torch.tensor([[0, 1], [1, 0]])
    stale_loss, _ = compute_reinforce_actor_loss(
        values,
        lambda_returns,
        continues,
        logits,
        actions,
        1.0,
        1.0,
        actor_entropy_coef=0.0,
        normalize_advantages=False,
        sample_mask=mask,
    )
    current_loss, _ = compute_reinforce_actor_loss(
        values,
        lambda_returns,
        continues,
        logits,
        actions,
        scale,
        1.0,
        actor_entropy_coef=0.0,
        normalize_advantages=False,
        sample_mask=mask,
    )

    assert current_loss.item() == pytest.approx(stale_loss.item() / scale)
