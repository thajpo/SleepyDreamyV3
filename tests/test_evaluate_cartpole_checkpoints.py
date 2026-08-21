import pytest
import torch

from scripts.evaluate_cartpole_checkpoints import (
    select_posterior_latent,
    select_policy_action,
    summarize_returns,
)


def test_summarize_returns_preserves_range_and_solved_fraction():
    summary = summarize_returns([100.0, 500.0, 500.0, 300.0])

    assert summary == {
        "episodes": 4,
        "mean_return": 350.0,
        "median_return": 400.0,
        "min_return": 100.0,
        "max_return": 500.0,
        "solved_fraction": 0.5,
    }


def test_summarize_returns_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        summarize_returns([])


def test_select_policy_action_supports_argmax_and_reproducible_sampling():
    logits = torch.tensor([[0.0, 0.0]]).expand(12, -1)
    first = torch.Generator().manual_seed(19)
    second = torch.Generator().manual_seed(19)

    assert select_policy_action(logits[:1], policy_mode="argmax").item() == 0
    assert torch.equal(
        select_policy_action(logits, policy_mode="sample", generator=first),
        select_policy_action(logits, policy_mode="sample", generator=second),
    )


def test_select_policy_action_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported policy mode"):
        select_policy_action(torch.zeros(1, 2), policy_mode="greedy")


def test_select_policy_action_uses_configured_unimix_for_sampling():
    logits = torch.tensor([[100.0, -100.0]]).expand(2_000, -1)
    actions = select_policy_action(
        logits,
        policy_mode="sample",
        generator=torch.Generator().manual_seed(23),
        actor_unimix=0.10,
    )

    alternative_fraction = (actions == 1).float().mean().item()
    assert alternative_fraction == pytest.approx(0.05, abs=0.015)


def test_select_posterior_latent_supports_mode_and_reproducible_sampling():
    logits = torch.tensor(
        [
            [[3.0, 1.0], [1.0, 3.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    first = torch.Generator().manual_seed(11)
    second = torch.Generator().manual_seed(11)

    torch.testing.assert_close(
        select_posterior_latent(logits[:1], latent_mode="argmax"),
        torch.tensor([[0, 1]]),
    )
    torch.testing.assert_close(
        select_posterior_latent(logits, latent_mode="sample", generator=first),
        select_posterior_latent(logits, latent_mode="sample", generator=second),
    )


def test_select_posterior_latent_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported latent mode"):
        select_posterior_latent(torch.zeros(1, 1, 2), latent_mode="mean")
