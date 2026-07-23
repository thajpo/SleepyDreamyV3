import gymnasium as gym
import pytest
import torch

from scripts.probe_cartpole_critic_supervision import (
    ReferenceValueHead,
    episode_split,
    trusted_remaining_return,
)


def test_episode_split_keeps_episodes_disjoint():
    episode_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    train, test = episode_split(episode_ids, test_fraction=0.25, seed=3)

    train_episodes = set(episode_ids[train].tolist())
    test_episodes = set(episode_ids[test].tolist())
    assert train_episodes
    assert test_episodes
    assert train_episodes.isdisjoint(test_episodes)


@pytest.mark.parametrize("test_fraction", [0.0, 1.0, -0.1, 1.1])
def test_episode_split_rejects_invalid_fraction(test_fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        episode_split(torch.tensor([0, 0, 1, 1]), test_fraction, seed=3)


def test_trusted_remaining_return_is_finite_and_positive():
    env = gym.make("CartPole-v1")
    try:
        value = trusted_remaining_return(
            env,
            state=torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy(),
            gamma=0.997,
            target_policy="heuristic",
            max_steps=50,
        )
    finally:
        env.close()

    assert 1.0 < value <= 50.0


def test_reference_value_head_has_three_normalized_hidden_layers():
    head = ReferenceValueHead(d_in=7, d_hidden=16, d_out=5)

    assert sum(isinstance(layer, torch.nn.RMSNorm) for layer in head.mlp) == 3
    assert head(torch.randn(4, 7)).shape == (4, 5)
    assert torch.count_nonzero(head.mlp[-1].weight) == 0
    assert torch.count_nonzero(head.mlp[-1].bias) == 0
