from types import SimpleNamespace

import pytest
import torch

from dreamer.config import Config
from dreamer.models import (
    initialize_critic,
    initialize_world_model,
    symexp,
    symexp_twohot_bins,
    symlog,
    twohot_encode,
    twohot_expectation,
)
from dreamer.trainer.core import WorldModelTrainer


def test_twohot_encode():
    B = torch.arange(start=-20, end=21)
    B = symexp(B)

    bsz = 15
    x = torch.rand(bsz)

    weights = twohot_encode(x, B)

    # Check that the output shape is correct
    assert weights.shape == (bsz, len(B))
    # Check that the weights for each sample sum to 1
    assert torch.allclose(weights.sum(dim=1), torch.ones(bsz))


def test_symmetric_twohot_zero_logits_decode_exactly_to_zero():
    bins = symexp_twohot_bins(-20, 20, 255)
    prediction = twohot_expectation(torch.zeros(4, 255), bins)

    torch.testing.assert_close(bins, -bins.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(prediction, torch.zeros(4), rtol=0, atol=0)


def test_even_symmetric_twohot_bins_are_distinct_mirrored_pairs():
    bins = symexp_twohot_bins(-20, 20, 254)
    prediction = twohot_expectation(torch.zeros(4, 254), bins)

    assert torch.all(bins[1:] > bins[:-1])
    assert not torch.any(bins == 0)
    torch.testing.assert_close(bins, -bins.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(prediction, torch.zeros(4), rtol=0, atol=0)


def test_initialized_reward_and_value_heads_decode_to_exactly_zero():
    config = Config()
    critic = initialize_critic("cpu", config)
    _encoder, world_model = initialize_world_model("cpu", config, batch_size=2)
    feature_size = config.d_hidden * config.rnn_n_blocks + (
        config.num_latents * (config.d_hidden // 16)
    )
    features = torch.randn(2, feature_size)
    bins = symexp_twohot_bins(
        config.b_start, config.b_end, config.num_bins
    )

    reward = twohot_expectation(world_model.reward_predictor(features), bins)
    value = twohot_expectation(critic(features), bins)

    torch.testing.assert_close(reward, torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(value, torch.zeros(2), rtol=0, atol=0)


@pytest.mark.parametrize(("train_step", "resume_offset"), [(0, 0), (3, 36)])
def test_pacing_gate_funds_collection_until_next_update_is_affordable(
    monkeypatch, train_step, resume_offset
):
    trainer = object.__new__(WorldModelTrainer)
    trainer.train_step = train_step
    trainer.batch_size = 4
    trainer._resume_env_steps_offset = resume_offset
    budget_targets = []
    trainer.replay_buffer = SimpleNamespace(
        total_env_steps=4,
        allow_env_steps_until=lambda steps: budget_targets.append(steps),
    )
    trainer.config = SimpleNamespace(
        replay_burn_in=1,
        sequence_length=4,
        replay_ratio=1.0,
        action_repeat=1,
    )
    sleep_calls = []
    monkeypatch.setattr(
        "dreamer.trainer.core.time.sleep", lambda seconds: sleep_calls.append(seconds)
    )

    assert trainer.prevent_stale_training()
    assert budget_targets == [12]
    assert sleep_calls == [0.01]

    trainer.replay_buffer.total_env_steps = 12
    assert not trainer.prevent_stale_training()
    assert budget_targets == [12]


@pytest.mark.parametrize("num_bins", [8, 9])
def test_twohot_expectation_preserves_weighted_sum_and_gradients(num_bins):
    bins = symexp_twohot_bins(-3, 4, num_bins)
    logits = torch.linspace(-1, 1, 2 * num_bins).view(2, num_bins)
    logits.requires_grad_()

    prediction = twohot_expectation(logits, bins)
    expected = (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)

    torch.testing.assert_close(prediction, expected, rtol=1e-5, atol=1e-5)
    prediction.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
