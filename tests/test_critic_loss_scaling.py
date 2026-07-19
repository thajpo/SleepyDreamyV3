import pytest
import torch

from dreamer.trainer.forward import add_sequence_mean_auxiliary_loss


def test_auxiliary_critic_scale_survives_sequence_normalization():
    sequence_losses = torch.tensor([2.0, 4.0, 6.0, 8.0])
    auxiliary_loss = torch.tensor(10.0)

    total = add_sequence_mean_auxiliary_loss(
        sequence_losses.sum(),
        auxiliary_loss,
        scale=0.3,
        sequence_length=len(sequence_losses),
    )

    normalized = total / len(sequence_losses)
    assert normalized.item() == pytest.approx(
        sequence_losses.mean().item() + 0.3 * auxiliary_loss.item()
    )


def test_auxiliary_critic_scale_rejects_empty_sequence():
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        add_sequence_mean_auxiliary_loss(
            torch.tensor(1.0),
            torch.tensor(1.0),
            scale=0.3,
            sequence_length=0,
        )
