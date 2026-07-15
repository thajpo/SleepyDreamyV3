from dataclasses import dataclass, field

import pytest

from dreamer.main import ChildProcessError, join_training_processes


@dataclass
class FakeProcess:
    exitcode: int | None = 0
    alive: bool = False
    joins: list[float | None] = field(default_factory=list)
    terminated: bool = False

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return self.alive and not self.terminated

    def terminate(self) -> None:
        self.terminated = True


@dataclass
class FakeStopEvent:
    was_set: bool = False

    def set(self) -> None:
        self.was_set = True


def test_successful_children_complete_cleanly():
    trainer = FakeProcess()
    collector = FakeProcess()
    stop_event = FakeStopEvent()

    join_training_processes(trainer, [collector], stop_event)

    assert stop_event.was_set
    assert trainer.joins == [None]
    assert collector.joins == [5.0]
    assert not collector.terminated


def test_trainer_failure_stops_collectors_and_propagates():
    trainer = FakeProcess(exitcode=7)
    collector = FakeProcess()
    stop_event = FakeStopEvent()

    with pytest.raises(ChildProcessError, match="trainer exited with code 7"):
        join_training_processes(trainer, [collector], stop_event)

    assert stop_event.was_set
    assert collector.joins == [5.0]


def test_collector_failure_propagates():
    stop_event = FakeStopEvent()

    with pytest.raises(ChildProcessError, match=r"collector\[0\] exited with code 3"):
        join_training_processes(
            FakeProcess(), [FakeProcess(exitcode=3)], stop_event
        )

    assert stop_event.was_set


def test_collector_that_ignores_shutdown_is_terminated_and_fails():
    collector = FakeProcess(exitcode=None, alive=True)

    with pytest.raises(ChildProcessError, match="did not stop within 0.1s"):
        join_training_processes(
            FakeProcess(), [collector], FakeStopEvent(), collector_timeout=0.1
        )

    assert collector.terminated
    assert collector.joins == [0.1, 0.1]
