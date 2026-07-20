import copy

import torch

from dreamer.trainer.forward import compute_replay_value_loss


def test_replay_value_loss_updates_observed_features_but_not_bootstrap_state():
    batch_size, time_steps, feature_size, num_bins = 2, 4, 3, 7
    posterior = torch.randn(
        batch_size,
        time_steps,
        feature_size,
        requires_grad=True,
    )
    critic = torch.nn.Linear(feature_size, num_bins)
    critic_ema = copy.deepcopy(critic)
    critic_ema.requires_grad_(False)
    rewards = torch.ones(time_steps, batch_size)
    is_last = torch.zeros(time_steps, batch_size)
    continues = torch.ones(time_steps, batch_size)
    mask = torch.ones(time_steps, batch_size)
    annotations = torch.zeros(time_steps, batch_size, requires_grad=True)
    bins = torch.linspace(-3.0, 3.0, num_bins)

    losses = compute_replay_value_loss(
        posterior,
        rewards,
        is_last,
        continues,
        mask,
        annotations,
        critic,
        critic_ema,
        bins,
        gamma=0.997,
        lam=0.95,
        slow_regularizer_scale=1.0,
    )
    losses.total.backward()

    assert posterior.grad is not None
    assert posterior.grad[:, :-1].abs().sum() > 0
    assert torch.equal(posterior.grad[:, -1], torch.zeros_like(posterior.grad[:, -1]))
    assert critic.weight.grad is not None
    assert critic.weight.grad.abs().sum() > 0
    assert annotations.grad is None
    assert all(parameter.grad is None for parameter in critic_ema.parameters())
