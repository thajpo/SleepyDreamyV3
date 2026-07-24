import multiprocessing as mp
import threading
from queue import Queue
from types import SimpleNamespace

import numpy as np

from dreamer.runtime.collector import collect_experiences


class _ActionSpace:
    def seed(self, _seed):
        return None

    def sample(self):
        return 0


class _FiveStepEnv:
    def __init__(self):
        self.action_space = _ActionSpace()
        self.step_count = 0

    def reset(self, *, seed):
        del seed
        self.step_count = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, _action):
        self.step_count += 1
        observation = np.full(2, self.step_count, dtype=np.float32)
        terminated = self.step_count == 5
        return observation, 1.0, terminated, False, {}

    def close(self):
        return None


def test_collector_publishes_preterminal_chunk_then_final_remainder(monkeypatch):
    monkeypatch.setattr(
        "dreamer.runtime.collector.create_env", lambda *_args, **_kwargs: _FiveStepEnv()
    )
    context = mp.get_context("spawn")
    data_queue = context.Queue(maxsize=8)
    stop_event = threading.Event()
    config = SimpleNamespace(
        use_pixels=False,
        environment_name="FakeEnv-v0",
        n_actions=2,
        action_repeat=1,
        seed=7,
        sequence_length=4,
        continuous_replay_delivery=True,
    )
    collector = threading.Thread(
        target=collect_experiences,
        args=(data_queue, Queue(), config, stop_event),
        kwargs={"collector_id": 3},
        daemon=True,
    )
    collector.start()
    try:
        first = data_queue.get(timeout=2.0)
        final = data_queue.get(timeout=2.0)
    finally:
        stop_event.set()
        collector.join(timeout=2.0)
        data_queue.close()

    assert not collector.is_alive()
    assert len(first[1]) == 4
    assert (first[7], first[8], first[9], first[10]) == (3, 1, False, 0)
    assert not first[4].any()
    assert len(final[1]) == 1
    assert (final[7], final[8], final[9], final[10]) == (3, 1, True, 4)
    assert final[4].tolist() == [True]

