import torch

from dreamer.trainer.forward import (
    calculate_replay_lambda_targets,
    calculate_replay_mc_targets,
)


def test_replay_lambda_targets_start_with_following_transition_reward():
    rewards = torch.tensor([[10.0], [20.0], [30.0]])
    continues = torch.tensor([[1.0], [1.0], [0.0]])
    annotations = torch.tensor([[100.0], [200.0], [300.0]])

    targets = calculate_replay_lambda_targets(
        rewards,
        continues,
        annotations,
        gamma=0.5,
        lam=0.0,
    )

    # Posterior row 0 is the state reached after reward 10. Its next-step
    # target begins at reward 20 and bootstraps from posterior row 1.
    assert torch.equal(targets, torch.tensor([[120.0], [30.0]]))


def test_replay_mc_targets_exclude_incoming_reward_and_last_state():
    rewards = torch.tensor([[10.0], [20.0], [30.0]])
    continues = torch.tensor([[1.0], [1.0], [0.0]])

    targets = calculate_replay_mc_targets(rewards, continues, gamma=1.0)

    assert torch.equal(targets, torch.tensor([[50.0], [30.0]]))
