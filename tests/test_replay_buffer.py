from queue import Queue
import time

import numpy as np
import pytest
import torch

from dreamer.runtime.replay_buffer import (
    EpisodeReplayBuffer,
    continuation_inclusion_weights,
)


def _episode(length: int, marker: float = 1.0):
    return (
        None,
        np.full((length, 2), marker, dtype=np.float32),
        np.zeros((length, 2), dtype=np.float32),
        np.arange(length, dtype=np.float32),
        np.zeros(length, dtype=np.bool_),
        np.zeros(length, dtype=np.bool_),
        length,
    )


def test_short_episode_is_padded_and_masked():
    replay = EpisodeReplayBuffer(
        data_queue=None,
        max_episodes=10,
        min_episodes=1,
        sequence_length=5,
        compute_future_returns=True,
    )
    replay.add_episode(_episode(3))

    batch = replay.sample_tensors(1, "cpu", recent_fraction=0.0)

    assert batch.states.shape == (1, 5, 2)
    assert torch.equal(batch.mask[0], torch.tensor([1, 1, 1, 0, 0]).float())
    assert torch.equal(batch.is_terminal[0, 3:], torch.ones(2, dtype=torch.bool))
    assert torch.equal(batch.future_returns[0, 3:], torch.zeros(2))
    assert torch.equal(
        batch.continue_weights[0], torch.tensor([1, 1, 1, 0, 0]).float()
    )
    assert replay.total_env_steps == 3


def test_continuation_weights_remove_window_edge_bias() -> None:
    episode_length = 7
    sequence_length = 3
    valid_starts = episode_length - sequence_length + 1
    aggregate = np.zeros(episode_length, dtype=np.float64)
    sampled_weights = []

    for start in range(valid_starts):
        weights = continuation_inclusion_weights(
            episode_length, sequence_length, start
        )
        aggregate[start : start + sequence_length] += weights
        sampled_weights.extend(weights.tolist())

    expected_multiplicity = sequence_length * valid_starts / episode_length
    assert np.allclose(aggregate, expected_multiplicity)
    assert np.mean(sampled_weights) == pytest.approx(1.0)
    assert continuation_inclusion_weights(
        episode_length, sequence_length, valid_starts - 1
    )[-1] == pytest.approx(expected_multiplicity)


def test_future_returns_start_after_each_post_transition_state():
    replay = EpisodeReplayBuffer(
        data_queue=None,
        max_episodes=10,
        min_episodes=1,
        sequence_length=4,
        gamma=0.5,
        compute_future_returns=True,
    )
    replay.add_episode(_episode(4))

    batch = replay.sample_tensors(1, "cpu", recent_fraction=0.0)

    # Row t is the state reached by reward[t], so its value begins at reward[t+1].
    assert torch.equal(
        batch.future_returns[0], torch.tensor([2.75, 3.5, 3.0, 0.0])
    )


def test_subsequence_return_includes_rewards_beyond_sample_window(monkeypatch):
    replay = EpisodeReplayBuffer(
        data_queue=None,
        max_episodes=10,
        min_episodes=1,
        sequence_length=2,
        gamma=1.0,
        compute_future_returns=True,
    )
    episode = list(_episode(5))
    episode[4][-1] = True
    replay.add_episode(tuple(episode))
    monkeypatch.setattr("dreamer.runtime.replay_buffer.random.randint", lambda *_: 1)

    sample = replay.sample(batch_size=1, recent_fraction=0.0)[0]

    # The second sampled row is episode row 2. Its target includes rewards 3 and
    # 4 even though neither future state is part of this two-row training slice.
    assert np.array_equal(sample[6], np.array([9.0, 7.0], dtype=np.float32))


def test_recent_only_sampling_uses_newest_episode():
    replay = EpisodeReplayBuffer(
        data_queue=None, max_episodes=10, min_episodes=1, sequence_length=2
    )
    for marker in range(5):
        replay.add_episode(_episode(2, marker=float(marker)))

    samples = replay.sample(batch_size=3, recent_fraction=1.0)

    assert all(np.allclose(sample[1], 4.0) for sample in samples)


def test_episode_selection_is_weighted_by_valid_sequence_starts(monkeypatch):
    replay = EpisodeReplayBuffer(
        data_queue=None, max_episodes=20, min_episodes=1, sequence_length=3
    )
    for _ in range(9):
        replay.add_episode(_episode(3, marker=0.0))
    replay.add_episode(_episode(6, marker=1.0))
    calls = []

    def choose(population, *, weights, k):
        calls.append((list(population), list(weights), k))
        return [population[-1]] * k

    monkeypatch.setattr("dreamer.runtime.replay_buffer.random.choices", choose)

    samples = replay.sample(batch_size=4, recent_fraction=0.5)

    assert calls == [
        ([8, 9], [1, 4], 2),
        (list(range(10)), [1] * 9 + [4], 2),
    ]
    assert all(np.allclose(sample[1], 1.0) for sample in samples)


def test_future_returns_are_not_stored_when_disabled():
    replay = EpisodeReplayBuffer(
        data_queue=None, max_episodes=10, min_episodes=1, sequence_length=2
    )
    replay.add_episode(_episode(2))

    batch = replay.sample_tensors(1, "cpu", recent_fraction=0.0)

    assert batch.future_returns is None
    assert replay.buffer[0][7] is None


def test_background_collection_allows_one_episode_of_budget_debt():
    queue = Queue()
    replay = EpisodeReplayBuffer(
        data_queue=queue,
        max_episodes=10,
        min_episodes=1,
        sequence_length=2,
        throttle_collection=True,
    )
    queue.put(_episode(3))
    queue.put(_episode(4))
    queue.put(_episode(2))
    replay.start()
    try:
        assert replay.ready_event.wait(timeout=1.0)
        assert replay.total_env_steps == 3

        replay.allow_env_steps(3)
        for _ in range(100):
            if replay.total_env_steps == 7:
                break
            time.sleep(0.01)
        assert replay.total_env_steps == 7

        # The four-step episode exceeded the three-step allowance by one step.
        # Collection remains stopped until learning first repays that debt and
        # then supplies any positive budget for another complete episode.
        replay.allow_env_steps(1)
        time.sleep(0.05)
        assert replay.total_env_steps == 7

        replay.allow_env_steps(0.1)
        for _ in range(100):
            if replay.total_env_steps == 9:
                break
            time.sleep(0.01)
        assert replay.total_env_steps == 9
    finally:
        replay.stop()


def test_collection_budget_rejects_negative_increment():
    replay = EpisodeReplayBuffer(throttle_collection=True)

    with pytest.raises(ValueError, match="non-negative"):
        replay.allow_env_steps(-1)
