import numpy as np
import torch

from dreamer.runtime.replay_buffer import EpisodeReplayBuffer


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
        data_queue=None, max_episodes=10, min_episodes=1, sequence_length=5
    )
    replay.add_episode(_episode(3))

    batch = replay.sample_tensors(1, "cpu", recent_fraction=0.0)

    assert batch.states.shape == (1, 5, 2)
    assert torch.equal(batch.mask[0], torch.tensor([1, 1, 1, 0, 0]).float())
    assert torch.equal(batch.is_terminal[0, 3:], torch.ones(2, dtype=torch.bool))
    assert torch.equal(batch.future_returns[0, 3:], torch.zeros(2))
    assert replay.total_env_steps == 3


def test_future_returns_start_after_each_post_transition_state():
    replay = EpisodeReplayBuffer(
        data_queue=None,
        max_episodes=10,
        min_episodes=1,
        sequence_length=4,
        gamma=0.5,
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
