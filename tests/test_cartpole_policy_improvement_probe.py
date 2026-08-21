import pytest
import torch

from scripts.probe_cartpole_policy_improvement import (
    confident_policy_label,
    deterministic_split,
)


def test_confident_policy_label_requires_separated_binary_values():
    label = confident_policy_label(
        torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.1])
    )
    assert label is not None
    assert label[0] == 1
    assert label[1] == pytest.approx(1.0)
    assert confident_policy_label(
        torch.tensor([1.0, 1.1]), torch.tensor([0.1, 0.1])
    ) is None


def test_confident_policy_label_rejects_nonbinary_input():
    with pytest.raises(ValueError, match="exactly two actions"):
        confident_policy_label(torch.ones(3), torch.ones(3))


def test_deterministic_split_is_disjoint_complete_and_repeatable():
    train_a, validation_a = deterministic_split(10, 0.2, 7)
    train_b, validation_b = deterministic_split(10, 0.2, 7)

    assert torch.equal(train_a, train_b)
    assert torch.equal(validation_a, validation_b)
    assert len(train_a) == 8
    assert len(validation_a) == 2
    assert sorted(torch.cat([train_a, validation_a]).tolist()) == list(range(10))
    assert set(train_a.tolist()).isdisjoint(validation_a.tolist())


@pytest.mark.parametrize(
    ("count", "fraction"), [(1, 0.2), (10, 0.0), (10, 1.0)]
)
def test_deterministic_split_rejects_invalid_contract(count, fraction):
    with pytest.raises(ValueError):
        deterministic_split(count, fraction, 7)
