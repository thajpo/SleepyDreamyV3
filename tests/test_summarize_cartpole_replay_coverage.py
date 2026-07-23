import argparse
from pathlib import Path

import pytest

from scripts.summarize_cartpole_replay_coverage import (
    parse_interval,
    read_mlflow_metric,
    summarize_run,
    summarize_values,
)


def _write_metric(metrics_dir: Path, name: str, rows: list[tuple[int, float]]) -> None:
    path = metrics_dir / name
    path.write_text("".join(f"1000 {value} {step}\n" for step, value in rows))


def test_read_metric_keeps_last_value_for_duplicate_step(tmp_path: Path):
    path = tmp_path / "metric"
    path.write_text("1000 1.0 25\n1001 2.0 25\n")

    assert read_mlflow_metric(path) == {25: 2.0}


def test_summarize_values_reports_ordered_endpoints_and_percentile():
    summary = summarize_values([(50, 3.0), (25, 1.0), (75, 2.0)])

    assert summary is not None
    assert summary["first"] == 1.0
    assert summary["last"] == 2.0
    assert summary["median"] == 2.0
    assert summary["p90"] == pytest.approx(2.8)


def test_summarize_run_derives_boundary_coverage_and_preserves_sparse_counts(
    tmp_path: Path,
):
    _write_metric(
        tmp_path,
        "research.cartpole.replay_abs_x.mean",
        [(25, 0.2), (50, 0.4)],
    )
    for label, values in {
        "0_0p5": [(25, 0.5), (50, 0.2)],
        "0p5_1p0": [(25, 0.2), (50, 0.2)],
        "1p0_1p5": [(25, 0.1), (50, 0.2)],
        "1p5_2p0": [(25, 0.1), (50, 0.2)],
        "2p0_plus": [(25, 0.1), (50, 0.2)],
    }.items():
        _write_metric(
            tmp_path,
            f"research.cartpole.replay_abs_x_fraction.{label}",
            values,
        )
    _write_metric(
        tmp_path,
        "research.cartpole.decoder_x_mse.2p0_plus",
        [(50, 4.0)],
    )

    result = summarize_run(tmp_path, [(0, 50), (50, 100)])

    assert result["time_series"][0][
        "derived.replay_abs_x_fraction.1p0_plus"
    ] == pytest.approx(0.3)
    first_metrics = result["intervals"]["0:50"]["metrics"]
    second_metrics = result["intervals"]["50:100"]["metrics"]
    assert first_metrics["research.cartpole.decoder_x_mse.2p0_plus"] is None
    assert second_metrics["research.cartpole.decoder_x_mse.2p0_plus"]["count"] == 1


def test_parse_interval_rejects_empty_or_reversed_ranges():
    assert parse_interval("2600:3000") == (2600, 3000)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_interval("3000:2600")


def test_summarize_run_rejects_missing_required_coverage_metrics(tmp_path: Path):
    _write_metric(
        tmp_path,
        "research.cartpole.replay_abs_x.mean",
        [(25, 0.2)],
    )

    with pytest.raises(FileNotFoundError, match="required replay telemetry"):
        summarize_run(tmp_path, [(0, 50)])
