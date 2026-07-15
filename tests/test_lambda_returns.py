import torch

from dreamer.models.dreaming import calculate_lambda_returns


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
