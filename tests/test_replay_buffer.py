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
    assert replay.total_env_steps == 3


def test_recent_only_sampling_uses_newest_episode():
    replay = EpisodeReplayBuffer(
        data_queue=None, max_episodes=10, min_episodes=1, sequence_length=2
    )
    for marker in range(5):
        replay.add_episode(_episode(2, marker=float(marker)))

    samples = replay.sample(batch_size=3, recent_fraction=1.0)

    assert all(np.allclose(sample[1], 4.0) for sample in samples)
