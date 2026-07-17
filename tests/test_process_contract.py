from dataclasses import dataclass, field

import pytest

from dreamer.main import ChildProcessError, join_training_processes


@dataclass
class FakeProcess:
    name: str = "process"
    exitcode: int | None = 0
    alive: bool = False
    joins: list[float | None] = field(default_factory=list)
    terminated: bool = False
    actions: list[str] | None = None

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)
        if self.actions is not None:
            self.actions.append(f"{self.name}.join")

    def is_alive(self) -> bool:
        return self.alive and not self.terminated

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15
        if self.actions is not None:
            self.actions.append(f"{self.name}.terminate")


@dataclass
class FakeStopEvent:
    name: str = "event"
    was_set: bool = False
    actions: list[str] | None = None

    def set(self) -> None:
        self.was_set = True
        if self.actions is not None:
            self.actions.append(f"{self.name}.set")

    def wait(self, timeout: float | None = None) -> bool:
        return self.was_set


def test_successful_children_complete_cleanly():
    actions: list[str] = []
    trainer = FakeProcess(name="trainer", actions=actions)
    collector = FakeProcess(name="collector", actions=actions)
    stop_event = FakeStopEvent(name="stop", actions=actions)
    training_done = FakeStopEvent(was_set=True)
    collectors_stopped = FakeStopEvent(name="collectors_stopped", actions=actions)

    join_training_processes(
        trainer,
        [collector],
        stop_event,
        training_done,
        collectors_stopped,
    )

    assert stop_event.was_set
    assert collectors_stopped.was_set
    assert trainer.joins == [5.0]
    assert collector.joins == [5.0]
    assert not collector.terminated
    assert actions.index("stop.set") < actions.index("collector.join")
    assert actions.index("collector.join") < actions.index("collectors_stopped.set")
    assert actions.index("collectors_stopped.set") < actions.index("trainer.join")


def test_trainer_failure_stops_collectors_and_propagates():
    trainer = FakeProcess(exitcode=7)
    collector = FakeProcess()
    stop_event = FakeStopEvent()

    with pytest.raises(ChildProcessError, match="trainer exited with code 7"):
        join_training_processes(
            trainer,
            [collector],
            stop_event,
            FakeStopEvent(),
            FakeStopEvent(),
        )

    assert stop_event.was_set
    assert collector.joins == [5.0]


def test_collector_failure_propagates():
    stop_event = FakeStopEvent()

    with pytest.raises(ChildProcessError, match=r"collector\[0\] exited with code 3"):
        join_training_processes(
            FakeProcess(exitcode=None, alive=True),
            [FakeProcess(exitcode=3)],
            stop_event,
            FakeStopEvent(),
            FakeStopEvent(),
        )

    assert stop_event.was_set


def test_collector_that_ignores_shutdown_is_terminated_and_fails():
    collector = FakeProcess(exitcode=None, alive=True)

    with pytest.raises(ChildProcessError, match="did not stop within 0.1s"):
        join_training_processes(
            FakeProcess(),
            [collector],
            FakeStopEvent(),
            FakeStopEvent(was_set=True),
            FakeStopEvent(),
            collector_timeout=0.1,
        )

    assert collector.terminated
    assert collector.joins == [0.1, 0.1]
