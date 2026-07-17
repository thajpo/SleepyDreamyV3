from queue import Queue

import torch

from dreamer.trainer.core import WorldModelTrainer


def _trainer_with_model_queues(count: int) -> WorldModelTrainer:
    trainer = object.__new__(WorldModelTrainer)
    trainer.model_queues = [Queue(maxsize=1) for _ in range(count)]
    trainer.actor = torch.nn.Linear(2, 2)
    trainer.encoder = torch.nn.Linear(2, 2)
    trainer.world_model = torch.nn.Linear(2, 2)
    return trainer


def test_model_update_is_published_to_every_collector_queue():
    trainer = _trainer_with_model_queues(3)

    trainer.send_models_to_collectors(step=7)

    updates = [queue.get_nowait() for queue in trainer.model_queues]
    assert [update["version"] for update in updates] == [7, 7, 7]
    for update in updates:
        assert set(update) == {"version", "actor", "encoder", "world_model"}


def test_busy_collector_keeps_one_pending_update_without_blocking_others():
    trainer = _trainer_with_model_queues(2)
    trainer.send_models_to_collectors(step=5)
    trainer.model_queues[0].get_nowait()

    trainer.send_models_to_collectors(step=10)

    assert trainer.model_queues[0].get_nowait()["version"] == 10
    assert trainer.model_queues[1].get_nowait()["version"] == 5
