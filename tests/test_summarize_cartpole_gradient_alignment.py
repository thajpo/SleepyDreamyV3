from pathlib import Path

import pytest

from scripts.summarize_cartpole_gradient_alignment import summarize_run


def _write_metric(metrics_dir: Path, name: str, rows: list[tuple[int, float]]) -> None:
    path = metrics_dir / name
    path.write_text("".join(f"1000 {value} {step}\n" for step, value in rows))


def _write_required_gradients(metrics_dir: Path) -> None:
    rows = [(100, 1.0), (200, 2.0), (300, 3.0), (400, 4.0)]
    for measure in ("wm_norm", "replay_norm", "replay_to_wm_norm"):
        _write_metric(
            metrics_dir,
            f"research.gradient_alignment.global.{measure}",
            rows,
        )


def test_summarizer_aligns_gradient_batches_to_latest_evaluation(tmp_path: Path):
    _write_required_gradients(tmp_path)
    _write_metric(
        tmp_path,
        "research.gradient_alignment.global.cosine",
        [(200, 0.2), (300, -0.4), (400, 0.1)],
    )
    _write_metric(
        tmp_path,
        "eval.episode_reward",
        [(150, 100.0), (200, 500.0), (300, 250.0), (400, 350.0)],
    )

    result = summarize_run(tmp_path)

    assert result["first_solved_step"] == 200
    assert result["categories"]["pre_solve"]["telemetry_batches"] == 1
    assert result["categories"]["solved"]["telemetry_batches"] == 1
    assert result["categories"]["degraded"]["telemetry_batches"] == 1
    assert result["categories"]["intermediate"]["telemetry_batches"] == 1
    degraded_cosine = result["categories"]["degraded"]["metrics"][
        "research.gradient_alignment.global.cosine"
    ]
    assert degraded_cosine["mean"] == pytest.approx(-0.4)


def test_summarizer_fails_closed_without_required_gradient_metrics(tmp_path: Path):
    _write_metric(tmp_path, "eval.episode_reward", [(100, 500.0)])

    with pytest.raises(FileNotFoundError, match="required gradient telemetry"):
        summarize_run(tmp_path)
