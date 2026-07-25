import random

import numpy as np
import torch

from dreamer.runtime.replay_buffer import EpisodeReplayBuffer
from dreamer.trainer.replay_evidence import (
    load_replay_evidence,
    save_replay_evidence,
)


def _episode(length: int, marker: float, episode_id: int):
    is_last = np.zeros(length, dtype=np.bool_)
    is_last[-1] = True
    return (
        None,
        np.full((length, 4), marker, dtype=np.float32),
        np.full((length, 2), marker, dtype=np.float32),
        np.full(length, marker, dtype=np.float32),
        is_last,
        np.zeros(length, dtype=np.bool_),
        length,
        0,
        episode_id,
    )


def test_state_evidence_is_read_only_deterministic_and_crosses_resets():
    replay = EpisodeReplayBuffer(
        data_queue=None,
        max_episodes=10,
        min_episodes=1,
        sequence_length=4,
        sequence_mode="stream",
        online_replay=True,
    )
    replay.add_episode(_episode(2, 10.0, 1))
    replay.add_episode(_episode(2, 20.0, 2))
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    queue_before = list(replay._online_queue)
    counters_before = (replay._online_samples, replay._sequence_samples)

    first = replay.sample_state_evidence(3, seed=17)
    second = replay.sample_state_evidence(3, seed=17)

    for name in first:
        assert np.array_equal(first[name], second[name])
    assert np.array_equal(first["states"][0, :, 0], [10, 10, 20, 20])
    assert np.array_equal(first["is_first"][0], [True, False, True, False])
    assert np.array_equal(first["start_episode_id"], [1, 1, 1])
    assert np.array_equal(first["start_offset"], [0, 0, 0])
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert list(replay._online_queue) == queue_before
    assert (replay._online_samples, replay._sequence_samples) == counters_before


def test_replay_evidence_round_trip_is_pickle_free(tmp_path):
    replay = EpisodeReplayBuffer(
        data_queue=None,
        max_episodes=10,
        min_episodes=1,
        sequence_length=4,
        sequence_mode="stream",
    )
    replay.add_episode(_episode(4, 3.0, 1))
    arrays = replay.sample_state_evidence(2, seed=19)
    destination = tmp_path / "replay_evidence_step_1.npz"

    saved = save_replay_evidence(
        destination,
        arrays,
        train_step=1,
        checkpoint_name="step_1",
    )
    loaded = load_replay_evidence(saved)

    assert loaded["states"].shape == (2, 4, 4)
    assert int(loaded["train_step"]) == 1
    assert str(loaded["checkpoint_name"]) == "step_1"
    assert not list(tmp_path.glob(".*.tmp"))
