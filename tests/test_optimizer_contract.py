from types import SimpleNamespace

import pytest
import torch

from dreamer.models import LaProp, adaptive_gradient_clipping
from dreamer.trainer.core import WorldModelTrainer
from dreamer.trainer.forward import replay_value_features


def _optimizer(parameter, lr=4e-5):
    return LaProp([parameter], lr=lr)


@pytest.mark.parametrize(
    ("step", "expected_scale"),
    [(0, 0.0), (500, 0.5), (1000, 1.0), (1500, 1.0)],
)
def test_reference_warmup_applies_one_scale_to_every_optimizer(
    step, expected_scale
):
    trainer = object.__new__(WorldModelTrainer)
    trainer.train_step = step
    trainer.config = SimpleNamespace(
        optimizer_warmup_steps=1000,
        lr_cosine_decay=False,
        lr_cosine_min_factor=0.1,
        max_train_steps=3500,
        wm_lr=4e-5,
        actor_lr=4e-5,
        critic_lr=4e-5,
    )
    parameters = [torch.nn.Parameter(torch.ones(())) for _ in range(3)]
    trainer.wm_optimizer = _optimizer(parameters[0])
    trainer.actor_optimizer = _optimizer(parameters[1])
    trainer.critic_optimizer = _optimizer(parameters[2])

    trainer.apply_lr_schedule()

    for optimizer in (
        trainer.wm_optimizer,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
    ):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            4e-5 * expected_scale
        )


def test_replay_value_representation_gradient_is_contract_gated():
    source = torch.tensor([1.0, -2.0], requires_grad=True)
    reference = replay_value_features(
        source, optimizer_contract="reference", critic_replay_scale=0.3
    )
    assert reference.requires_grad
    reference.square().sum().backward()
    torch.testing.assert_close(source.grad, 2.0 * source.detach())

    source.grad = None
    legacy = replay_value_features(
        source, optimizer_contract="legacy", critic_replay_scale=0.3
    )
    assert not legacy.requires_grad
    assert source.grad is None


def test_equal_rate_split_laprop_matches_one_joint_optimizer():
    initial = [torch.tensor([0.5, -1.0]), torch.tensor([1.5]), torch.tensor([-0.2])]
    split = [torch.nn.Parameter(value.clone()) for value in initial]
    joint = [torch.nn.Parameter(value.clone()) for value in initial]
    split_optimizers = [_optimizer(parameter) for parameter in split]
    joint_optimizer = LaProp(joint, lr=4e-5)

    for _ in range(3):
        for optimizer in split_optimizers:
            optimizer.zero_grad()
        joint_optimizer.zero_grad()

        split_loss = (
            (split[0] * split[1]).sum()
            + split[1].square().sum()
            + split[2].square().sum()
        )
        joint_loss = (
            (joint[0] * joint[1]).sum()
            + joint[1].square().sum()
            + joint[2].square().sum()
        )
        split_loss.backward()
        joint_loss.backward()

        for parameter in split:
            adaptive_gradient_clipping([parameter], clip_factor=0.3)
        adaptive_gradient_clipping(joint, clip_factor=0.3)
        for optimizer in split_optimizers:
            optimizer.step()
        joint_optimizer.step()

    for split_parameter, joint_parameter in zip(split, joint):
        torch.testing.assert_close(split_parameter, joint_parameter)
