import pytest
import torch

from dreamer.trainer.forward import calculate_continuation_balance


def test_disabled_continuation_balance_preserves_inclusion_weights():
    inclusion = torch.tensor([[2.0, 1.0]])
    result = calculate_continuation_balance(
        torch.tensor([[True, False]]),
        torch.ones(1, 2),
        inclusion,
        0.25,
        enabled=False,
        rate=0.01,
    )

    weights, next_ema, batch_fraction, terminal_scale, live_scale = result
    assert torch.equal(weights, inclusion)
    assert next_ema == 0.25
    assert batch_fraction == pytest.approx(2.0 / 3.0)
    assert terminal_scale == 1.0
    assert live_scale == 1.0


def test_continuation_prevalence_uses_masks_and_inclusion_weights():
    terminals = torch.tensor([[True, False, True], [False, True, False]])
    masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    inclusion = torch.tensor([[3.0, 1.0, 100.0], [2.0, 100.0, 4.0]])

    _, next_ema, batch_fraction, _, _ = calculate_continuation_balance(
        terminals,
        masks,
        inclusion,
        0.5,
        enabled=True,
        rate=0.1,
    )

    assert batch_fraction == pytest.approx(3.0 / 10.0)
    assert next_ema == pytest.approx(0.48)


def test_continuation_balance_equalizes_class_mass_at_matching_prevalence():
    terminals = torch.tensor([[True, False, False, False]])
    masks = torch.ones(1, 4)
    inclusion = torch.ones(1, 4)

    weights, next_ema, _, terminal_scale, live_scale = (
        calculate_continuation_balance(
            terminals,
            masks,
            inclusion,
            0.25,
            enabled=True,
            rate=0.01,
        )
    )

    assert next_ema == pytest.approx(0.25)
    assert terminal_scale == pytest.approx(2.0)
    assert live_scale == pytest.approx(2.0 / 3.0)
    assert weights[terminals].sum().item() == pytest.approx(2.0)
    assert weights[~terminals].sum().item() == pytest.approx(2.0)
    assert weights.mean().item() == pytest.approx(1.0)


def test_continuation_balance_first_update_is_not_debiased():
    _, next_ema, _, _, _ = calculate_continuation_balance(
        torch.zeros(1, 4, dtype=torch.bool),
        torch.ones(1, 4),
        torch.ones(1, 4),
        0.5,
        enabled=True,
        rate=0.01,
    )

    assert next_ema == pytest.approx(0.495)
