import pytest
import torch

from dreamer.trainer.forward import (
    add_sequence_mean_auxiliary_loss,
    normalize_sequence_losses,
)


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


def test_sequence_objectives_and_gradients_are_reduced_to_means():
    wm_parameter = torch.tensor(2.0, requires_grad=True)
    actor_parameter = torch.tensor(3.0, requires_grad=True)
    critic_parameter = torch.tensor(5.0, requires_grad=True)
    replay_parameter = torch.tensor(7.0, requires_grad=True)

    wm_steps = torch.stack([wm_parameter * factor for factor in (1.0, 2.0, 3.0)])
    actor_starts = torch.stack(
        [actor_parameter * factor for factor in (2.0, 4.0)]
    )
    critic_starts = torch.stack(
        [critic_parameter * factor for factor in (3.0, 6.0)]
    )
    replay_auxiliary = replay_parameter * 11.0
    critic_sum = add_sequence_mean_auxiliary_loss(
        critic_starts.sum(), replay_auxiliary, scale=0.3, sequence_length=2
    )
    replay_representation_sum = 2 * 0.3 * replay_auxiliary

    wm, actor, critic, replay_representation = normalize_sequence_losses(
        wm_steps.sum(),
        actor_starts.sum(),
        critic_sum,
        replay_length=3,
        imagination_starts=2,
        replay_representation_loss=replay_representation_sum,
    )

    assert wm.item() == pytest.approx(wm_steps.mean().item())
    assert actor.item() == pytest.approx(actor_starts.mean().item())
    assert critic.item() == pytest.approx(
        critic_starts.mean().item() + 0.3 * replay_auxiliary.item()
    )
    assert replay_representation is not None
    assert replay_representation.item() == pytest.approx(
        0.3 * replay_auxiliary.item()
    )

    (wm + actor + critic + replay_representation).backward()
    assert wm_parameter.grad.item() == pytest.approx(2.0)
    assert actor_parameter.grad.item() == pytest.approx(3.0)
    assert critic_parameter.grad.item() == pytest.approx(4.5)
    assert replay_parameter.grad.item() == pytest.approx(6.6)


@pytest.mark.parametrize(
    ("replay_length", "imagination_starts", "message"),
    [(0, 1, "replay_length"), (1, 0, "imagination_starts")],
)
def test_sequence_loss_reduction_rejects_empty_axes(
    replay_length, imagination_starts, message
):
    with pytest.raises(ValueError, match=message):
        normalize_sequence_losses(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            replay_length=replay_length,
            imagination_starts=imagination_starts,
        )
