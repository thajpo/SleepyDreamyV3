#!/usr/bin/env python3
"""Align CartPole gradient diagnostics with deterministic evaluations."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

if __package__:
    from scripts.summarize_cartpole_replay_coverage import (
        read_mlflow_metric,
        summarize_values,
    )
else:
    from summarize_cartpole_replay_coverage import (  # type: ignore[import-not-found]
        read_mlflow_metric,
        summarize_values,
    )


GROUPS = ("global", "encoder", "recurrent", "posterior")
MEASURES = (
    "wm_norm",
    "replay_norm",
    "replay_to_wm_norm",
    "cosine",
    "shared_numel",
)
METRICS = tuple(
    f"research.gradient_alignment.{group}.{measure}"
    for group in GROUPS
    for measure in MEASURES
)


def _latest_evaluation(
    evaluations: dict[int, float], step: int
) -> tuple[int, float] | None:
    evaluation_steps = sorted(evaluations)
    index = bisect.bisect_right(evaluation_steps, step) - 1
    if index < 0:
        return None
    evaluation_step = evaluation_steps[index]
    return evaluation_step, evaluations[evaluation_step]


def summarize_run(
    metrics_dir: Path,
    solved_threshold: float = 450.0,
    degraded_threshold: float = 300.0,
) -> dict[str, object]:
    """Summarize gradient metrics before, during, and after solved behavior."""
    metrics = {name: read_mlflow_metric(metrics_dir / name) for name in METRICS}
    evaluations = read_mlflow_metric(metrics_dir / "eval.episode_reward")
    required = (
        "research.gradient_alignment.global.wm_norm",
        "research.gradient_alignment.global.replay_norm",
        "research.gradient_alignment.global.replay_to_wm_norm",
    )
    missing = [name for name in required if not metrics[name]]
    if missing:
        raise FileNotFoundError(
            "required gradient telemetry is missing: " + ", ".join(missing)
        )
    if not evaluations:
        raise FileNotFoundError("required evaluation metric is missing")

    solved_steps = [
        step for step, value in evaluations.items() if value >= solved_threshold
    ]
    first_solved_step = min(solved_steps) if solved_steps else None
    telemetry_steps = sorted(metrics[required[0]])
    categories: dict[str, list[dict[str, float | int | str | None]]] = {
        "pre_solve": [],
        "solved": [],
        "degraded": [],
        "intermediate": [],
    }
    time_series: list[dict[str, float | int | str | None]] = []

    for step in telemetry_steps:
        latest = _latest_evaluation(evaluations, step)
        evaluation_step = latest[0] if latest else None
        evaluation_value = latest[1] if latest else None
        if first_solved_step is None or step < first_solved_step:
            category = "pre_solve"
        elif evaluation_value is not None and evaluation_value >= solved_threshold:
            category = "solved"
        elif evaluation_value is not None and evaluation_value < degraded_threshold:
            category = "degraded"
        else:
            category = "intermediate"

        row: dict[str, float | int | str | None] = {
            "step": step,
            "category": category,
            "latest_eval_step": evaluation_step,
            "latest_eval_reward": evaluation_value,
        }
        for name, metric_values in metrics.items():
            row[name] = metric_values.get(step)
        categories[category].append(row)
        time_series.append(row)

    summaries: dict[str, object] = {}
    for category, rows in categories.items():
        category_metrics: dict[str, object] = {}
        for name in METRICS:
            summary_values: list[tuple[int, float]] = []
            for row in rows:
                step = row["step"]
                value = row[name]
                if not isinstance(step, int):
                    raise TypeError("gradient telemetry step must be an integer")
                if isinstance(value, (float, int)):
                    summary_values.append((step, float(value)))
            category_metrics[name] = summarize_values(summary_values)
        summaries[category] = {
            "telemetry_batches": len(rows),
            "metrics": category_metrics,
        }

    return {
        "metrics_dir": str(metrics_dir.resolve()),
        "solved_threshold": solved_threshold,
        "degraded_threshold": degraded_threshold,
        "first_solved_step": first_solved_step,
        "categories": summaries,
        "time_series": time_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--solved-threshold", type=float, default=450.0)
    parser.add_argument("--degraded-threshold", type=float, default=300.0)
    args = parser.parse_args()
    summary = summarize_run(
        args.metrics_dir,
        solved_threshold=args.solved_threshold,
        degraded_threshold=args.degraded_threshold,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
