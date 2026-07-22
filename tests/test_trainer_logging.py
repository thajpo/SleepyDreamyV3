from types import SimpleNamespace

import pytest
import torch

from dreamer.trainer.logging import (
    create_step_metrics,
    log_progress,
    log_step_metrics,
)


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log_scalars(self, metrics, step):
        self.records.append((metrics, step))


def test_step_logging_does_not_reduce_mean_totals_twice():
    logger = RecordingLogger()
    metrics = create_step_metrics(torch.device("cpu"), do_log_images=False)

    log_step_metrics(
        logger,
        metrics,
        total_wm_loss=torch.tensor(12.0),
        total_actor_loss=torch.tensor(8.0),
        total_critic_loss=torch.tensor(4.0),
        sequence_length=4,
        step=0,
        config=SimpleNamespace(environment_name="Dummy-v0"),
        has_pixel_obs=False,
        has_vector_obs=False,
        log_every=1,
        image_log_every=100,
        log_profile="lean",
    )

    logged, step = logger.records[0]
    assert step == 0
    assert logged["loss/wm/total"] == 12.0
    assert logged["loss/actor/total"] == 8.0
    assert logged["loss/critic/total"] == 4.0


def test_step_logging_records_actor_action_support():
    logger = RecordingLogger()
    metrics = create_step_metrics(torch.device("cpu"), do_log_images=False)
    metrics.actor_min_action_probability.append(torch.tensor(0.05))
    metrics.actor_non_modal_probability.append(torch.tensor(0.08))
    metrics.actor_non_modal_sample_fraction.append(torch.tensor(0.10))

    log_step_metrics(
        logger,
        metrics,
        total_wm_loss=torch.tensor(1.0),
        total_actor_loss=torch.tensor(1.0),
        total_critic_loss=torch.tensor(1.0),
        sequence_length=1,
        step=0,
        config=SimpleNamespace(environment_name="Dummy-v0"),
        has_pixel_obs=False,
        has_vector_obs=False,
        log_every=1,
        image_log_every=100,
        log_profile="lean",
    )

    logged, _step = logger.records[0]
    assert logged["actor/support/min_action_probability_mean"] == pytest.approx(0.05)
    assert logged["actor/support/non_modal_probability_mean"] == pytest.approx(0.08)
    assert logged["actor/support/non_modal_sample_fraction"] == pytest.approx(0.10)


def test_progress_logging_does_not_reduce_mean_totals_twice(capsys):
    logger = RecordingLogger()

    log_progress(
        logger,
        step=10,
        max_steps=20,
        total_wm_loss=torch.tensor(12.0),
        total_actor_loss=torch.tensor(8.0),
        total_critic_loss=torch.tensor(4.0),
        seq_len=4,
        steps_per_sec=2.0,
        env_steps=100,
        episodes_added=5,
        avg_ep_len=20.0,
        elapsed_total=5.0,
    )

    output = capsys.readouterr().out
    assert "WM: 12.0000" in output
    assert "Actor: 8.0000" in output
    assert "Critic: 4.0000" in output
