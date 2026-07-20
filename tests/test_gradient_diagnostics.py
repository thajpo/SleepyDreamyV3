import copy

import torch

from dreamer.trainer.gradient_diagnostics import measure_gradient_alignment


def _losses(encoder, critic, inputs, targets):
    features = encoder(inputs)
    world_model_loss = features.square().mean()
    replay_training_loss = (critic(features.detach()) - targets).square().mean()
    replay_representation_loss = (critic(features) - targets).square().mean()
    return world_model_loss, replay_training_loss, replay_representation_loss


def test_gradient_diagnostic_is_read_only_and_preserves_training_gradients():
    torch.manual_seed(7)
    encoder = torch.nn.Linear(3, 2, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 1)
    control_encoder = copy.deepcopy(encoder)
    control_critic = copy.deepcopy(critic)
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    control_encoder_optimizer = torch.optim.SGD(
        control_encoder.parameters(), lr=0.1
    )
    control_critic_optimizer = torch.optim.SGD(
        control_critic.parameters(), lr=0.1
    )

    world_model_loss, replay_training_loss, replay_representation_loss = _losses(
        encoder, critic, inputs, targets
    )
    metrics = measure_gradient_alignment(
        world_model_loss,
        replay_representation_loss,
        [
            (f"encoder.{name}", parameter)
            for name, parameter in encoder.named_parameters()
        ],
    )
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert metrics["research/gradient_alignment/global/wm_norm"] > 0.0
    assert metrics["research/gradient_alignment/global/replay_norm"] > 0.0
    assert -1.0 <= metrics["research/gradient_alignment/global/cosine"] <= 1.0

    world_model_loss.backward()
    replay_training_loss.backward()

    control_wm_loss, control_replay_loss, _ = _losses(
        control_encoder, control_critic, inputs, targets
    )
    control_wm_loss.backward()
    control_replay_loss.backward()

    for parameter, control in zip(
        encoder.parameters(), control_encoder.parameters()
    ):
        torch.testing.assert_close(parameter.grad, control.grad)
    for parameter, control in zip(
        critic.parameters(), control_critic.parameters()
    ):
        torch.testing.assert_close(parameter.grad, control.grad)

    encoder_optimizer.step()
    critic_optimizer.step()
    control_encoder_optimizer.step()
    control_critic_optimizer.step()
    for parameter, control in zip(
        encoder.parameters(), control_encoder.parameters()
    ):
        torch.testing.assert_close(parameter, control)
    for parameter, control in zip(
        critic.parameters(), control_critic.parameters()
    ):
        torch.testing.assert_close(parameter, control)
