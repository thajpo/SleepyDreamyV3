import torch

from dreamer.trainer.forward import (
    calculate_replay_lambda_targets,
    calculate_replay_pair_mask,
)


def test_replay_lambda_targets_start_with_following_transition_reward():
    rewards = torch.tensor([[10.0], [20.0], [30.0]])
    is_last = torch.tensor([[False], [False], [True]])
    is_terminal = torch.tensor([[False], [False], [True]])
    annotations = torch.tensor([[100.0], [200.0], [300.0]])

    targets = calculate_replay_lambda_targets(
        rewards,
        is_last,
        is_terminal,
        annotations,
        gamma=0.5,
        lam=0.0,
    )

    # Posterior row 0 is the state reached after reward 10. Its next-step
    # target begins at reward 20 and bootstraps from posterior row 1.
    assert torch.equal(targets, torch.tensor([[120.0], [30.0]]))


def test_replay_lambda_targets_bootstrap_truncation_but_not_terminal():
    rewards = torch.tensor([[0.0], [2.0]])
    annotations = torch.tensor([[10.0], [20.0]])
    is_last = torch.tensor([[False], [True]])

    truncated = calculate_replay_lambda_targets(
        rewards,
        is_last,
        torch.tensor([[False], [False]]),
        annotations,
        gamma=0.5,
        lam=0.95,
    )
    terminated = calculate_replay_lambda_targets(
        rewards,
        is_last,
        torch.tensor([[False], [True]]),
        annotations,
        gamma=0.5,
        lam=0.95,
    )

    assert torch.equal(truncated, torch.tensor([[12.0]]))
    assert torch.equal(terminated, torch.tensor([[2.0]]))


def test_replay_lambda_targets_preserve_ordinary_lambda_recursion():
    targets = calculate_replay_lambda_targets(
        rewards=torch.tensor([[0.0], [2.0], [3.0]]),
        is_last=torch.zeros(3, 1, dtype=torch.bool),
        is_terminal=torch.zeros(3, 1, dtype=torch.bool),
        value_annotations=torch.tensor([[0.0], [10.0], [20.0]]),
        gamma=0.5,
        lam=0.5,
    )

    assert torch.equal(targets, torch.tensor([[7.75], [13.0]]))


def test_replay_pair_mask_excludes_padding_and_cross_episode_pairs():
    mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    is_last = torch.tensor([[False], [True], [False], [True]])

    pair_mask = calculate_replay_pair_mask(mask, is_last)

    assert torch.equal(pair_mask, torch.tensor([[1.0], [0.0], [0.0]]))
