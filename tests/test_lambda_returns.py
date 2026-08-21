import torch

from dreamer.models.dreaming import (
    calculate_lambda_returns,
    learned_continue_discount,
)
from dreamer.trainer.forward import select_critic_target_values


def test_discount_is_not_applied_twice_to_discounted_continuation():
    assert learned_continue_discount(gamma=0.997, contdisc=True) == 1.0
    assert learned_continue_discount(gamma=0.997, contdisc=False) == 0.997


def test_lambda_one_matches_discounted_monte_carlo_returns():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros_like(rewards)
    continues = torch.ones_like(rewards)

    returns = calculate_lambda_returns(
        rewards,
        values,
        continues,
        gamma=1.0,
        lam=1.0,
        num_dream_steps=3,
        continues_are_logits=False,
    )

    assert torch.equal(returns, torch.tensor([[6.0], [5.0], [3.0]]))


def test_lambda_zero_uses_next_step_value_alignment():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    continues = torch.ones_like(rewards)

    returns = calculate_lambda_returns(
        rewards,
        values,
        continues,
        gamma=1.0,
        lam=0.0,
        num_dream_steps=3,
        continues_are_logits=False,
    )

    assert torch.equal(returns, torch.tensor([[21.0], [32.0], [33.0]]))


def test_online_critic_target_is_detached_from_value_graph():
    online = torch.tensor([1.0], requires_grad=True)
    slow = torch.tensor([2.0])

    target = select_critic_target_values(online, slow, use_slow_target=False)

    assert torch.equal(target, torch.tensor([1.0]))
    assert target.requires_grad is False


def test_historical_slow_critic_target_remains_selectable():
    online = torch.tensor([1.0])
    slow = torch.tensor([2.0], requires_grad=True)

    target = select_critic_target_values(online, slow, use_slow_target=True)

    assert torch.equal(target, torch.tensor([2.0]))
    assert target.requires_grad is False
