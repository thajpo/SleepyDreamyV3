import pytest
import torch

from dreamer.trainer.logging import (
    StepMetrics,
    summarize_cartpole_replay_state_metrics,
)


def test_cartpole_replay_metrics_mask_padding_and_split_position_bins():
    metrics = StepMetrics(
        replay_states=[
            torch.tensor(
                [
                    [0.25, 1.0, 0.1, 2.0],
                    [0.75, 2.0, 0.2, 3.0],
                    [1.25, 3.0, 0.3, 4.0],
                    [1.75, 4.0, 0.4, 5.0],
                    [2.25, 5.0, 0.5, 6.0],
                    [9.00, 9.0, 9.0, 9.0],
                ]
            )
        ],
        replay_state_reconstructions=[torch.zeros(6, 4)],
        replay_state_masks=[torch.tensor([1, 1, 1, 1, 1, 0])],
    )

    summary = summarize_cartpole_replay_state_metrics(metrics)

    assert summary["research/cartpole/replay_abs_x/mean"] == 1.25
    assert summary["research/cartpole/replay_abs_x/max"] == 2.25
    assert summary["research/cartpole/decoder_mse/x"] == pytest.approx(2.0625)
    assert summary["research/cartpole/decoder_mse/x_dot"] == 11.0
    for label in ("0_0p5", "0p5_1p0", "1p0_1p5", "1p5_2p0", "2p0_plus"):
        assert summary[
            f"research/cartpole/replay_abs_x_fraction/{label}"
        ] == pytest.approx(0.2)
    assert summary["research/cartpole/decoder_x_mse/2p0_plus"] == pytest.approx(
        2.25**2
    )


def test_cartpole_replay_metrics_require_valid_four_component_rows():
    assert summarize_cartpole_replay_state_metrics(StepMetrics()) == {}
    metrics = StepMetrics(
        replay_states=[torch.ones(2, 3)],
        replay_state_reconstructions=[torch.zeros(2, 3)],
        replay_state_masks=[torch.ones(2)],
    )
    assert summarize_cartpole_replay_state_metrics(metrics) == {}
