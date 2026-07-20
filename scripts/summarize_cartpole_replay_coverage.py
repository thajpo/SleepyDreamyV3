#!/usr/bin/env python3
"""Summarize prospective CartPole replay telemetry from an MLflow run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TELEMETRY_METRICS = (
    "research.cartpole.replay_abs_x.mean",
    "research.cartpole.replay_abs_x.p90",
    "research.cartpole.replay_abs_x.max",
    "research.cartpole.replay_abs_x_fraction.0_0p5",
    "research.cartpole.replay_abs_x_fraction.0p5_1p0",
    "research.cartpole.replay_abs_x_fraction.1p0_1p5",
    "research.cartpole.replay_abs_x_fraction.1p5_2p0",
    "research.cartpole.replay_abs_x_fraction.2p0_plus",
    "research.cartpole.decoder_mse.x",
    "research.cartpole.decoder_mse.x_dot",
    "research.cartpole.decoder_mse.theta",
    "research.cartpole.decoder_mse.theta_dot",
    "research.cartpole.decoder_x_mse.0_0p5",
    "research.cartpole.decoder_x_mse.0p5_1p0",
    "research.cartpole.decoder_x_mse.1p0_1p5",
    "research.cartpole.decoder_x_mse.1p5_2p0",
    "research.cartpole.decoder_x_mse.2p0_plus",
    "actor.entropy.mean",
    "eval.episode_reward",
)


def read_mlflow_metric(path: Path) -> dict[int, float]:
    """Read an MLflow filesystem metric, keeping its final value per step."""
    values: dict[int, float] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        _timestamp, value, step = line.split()
        values[int(step)] = float(value)
    return values


def percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: list[tuple[int, float]]) -> dict[str, float | int] | None:
    """Summarize step/value pairs without hiding sparse metrics."""
    if not values:
        return None
    ordered = sorted(values)
    scalars = [value for _step, value in ordered]
    return {
        "count": len(scalars),
        "mean": statistics.fmean(scalars),
        "median": statistics.median(scalars),
        "p90": percentile(scalars, 0.9),
        "min": min(scalars),
        "max": max(scalars),
        "first": scalars[0],
        "last": scalars[-1],
    }


def parse_interval(value: str) -> tuple[int, int]:
    """Parse a half-open START:END train-step interval."""
    try:
        start_text, end_text = value.split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("interval must be START:END") from exc
    if start >= end:
        raise argparse.ArgumentTypeError("interval START must be less than END")
    return start, end


def _row_step(row: dict[str, float | int | None]) -> int:
    step = row["step"]
    if not isinstance(step, int):
        raise TypeError("telemetry row step must be an integer")
    return step


def summarize_run(
    metrics_dir: Path, intervals: list[tuple[int, int]]
) -> dict[str, object]:
    """Build interval summaries and an aligned telemetry time series."""
    metrics = {
        name: read_mlflow_metric(metrics_dir / name) for name in TELEMETRY_METRICS
    }
    fraction_prefix = "research.cartpole.replay_abs_x_fraction."
    required_names = (
        "research.cartpole.replay_abs_x.mean",
        *(f"{fraction_prefix}{label}" for label in (
            "0_0p5",
            "0p5_1p0",
            "1p0_1p5",
            "1p5_2p0",
            "2p0_plus",
        )),
    )
    missing = [name for name in required_names if not metrics[name]]
    if missing:
        raise FileNotFoundError(
            "required replay telemetry is missing: " + ", ".join(missing)
        )
    telemetry_steps = sorted(
        metrics["research.cartpole.replay_abs_x.mean"].keys()
    )

    time_series: list[dict[str, float | int | None]] = []
    for step in telemetry_steps:
        row: dict[str, float | int | None] = {"step": step}
        for name, metric_values in metrics.items():
            if name != "eval.episode_reward":
                row[name] = metric_values.get(step)
        row["derived.replay_abs_x_fraction.1p0_plus"] = sum(
            metrics[f"{fraction_prefix}{label}"][step]
            for label in ("1p0_1p5", "1p5_2p0", "2p0_plus")
        )
        row["derived.replay_abs_x_fraction.1p5_plus"] = sum(
            metrics[f"{fraction_prefix}{label}"][step]
            for label in ("1p5_2p0", "2p0_plus")
        )
        time_series.append(row)

    interval_summaries: dict[str, object] = {}
    for start, end in intervals:
        label = f"{start}:{end}"
        rows = [row for row in time_series if start <= _row_step(row) < end]
        summaries: dict[str, object] = {
            "telemetry_batches": len(rows),
            "metrics": {},
        }
        interval_metrics = summaries["metrics"]
        assert isinstance(interval_metrics, dict)
        row_names = set().union(*(row.keys() for row in rows)) if rows else set()
        for name in sorted(row_names - {"step"}):
            interval_values: list[tuple[int, float]] = []
            for row in rows:
                value = row.get(name)
                if value is not None:
                    interval_values.append((_row_step(row), float(value)))
            interval_metrics[name] = summarize_values(interval_values)
        interval_metrics["eval.episode_reward"] = summarize_values(
            [
                (step, value)
                for step, value in metrics["eval.episode_reward"].items()
                if start <= step < end
            ]
        )
        interval_summaries[label] = summaries

    return {
        "metrics_dir": str(metrics_dir.resolve()),
        "interval_semantics": "half-open [start, end)",
        "intervals": interval_summaries,
        "time_series": time_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--interval",
        action="append",
        type=parse_interval,
        default=[],
        help="Half-open train-step interval START:END; repeat as needed.",
    )
    args = parser.parse_args()
    intervals = args.interval or [(0, 2600), (2600, 3000), (3000, 3501)]
    summary = summarize_run(args.metrics_dir, intervals)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
