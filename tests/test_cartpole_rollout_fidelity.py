import pytest
import torch

from scripts.probe_cartpole_rollout_fidelity import (
    discounted_return_to_go,
    optional_masked_mean,
    return_error_components,
)


def test_discounted_return_to_go_aligns_rewards():
    returns = discounted_return_to_go([1.0, 2.0, 3.0], gamma=0.5)

    assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_return_error_decomposition_is_exact():
    components = return_error_components(
        model_prefix=3.8,
        discounted_model_reward_prefix=4.0,
        predicted_discount=0.7,
        prior_bootstrap=4.2,
        actual_prefix=3.0,
        actual_discount=0.5,
        posterior_value=6.0,
        target_return=5.0,
    )

    decomposed = sum(
        components[key]
        for key in (
            "oracle_error",
            "reward_error",
            "continuation_prefix_error",
            "final_discount_error",
            "critic_transport_error",
        )
    )

    assert components["full_error"] == pytest.approx(decomposed)
    assert components["rollout_error"] == pytest.approx(
        components["reward_error"]
        + components["continuation_prefix_error"]
        + components["final_discount_error"]
        + components["critic_transport_error"]
    )
    assert components["prefix_error"] == pytest.approx(
        components["reward_error"] + components["continuation_prefix_error"]
    )
    assert components["bootstrap_transport_error"] == pytest.approx(
        components["final_discount_error"]
        + components["critic_transport_error"]
    )


def test_optional_masked_mean_marks_empty_cohort_unavailable():
    values = torch.tensor([1.0, 3.0])

    assert optional_masked_mean(values, torch.tensor([True, False])) == 1.0
    assert optional_masked_mean(values, torch.tensor([False, False])) is None
