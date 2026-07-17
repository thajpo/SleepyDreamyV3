import csv
import json
from pathlib import Path

from dreamer.run_registry import generate_registry


def _config(**updates):
    values = {
        "environment_name": "CartPole-v1",
        "use_pixels": False,
        "n_observations": 4,
        "seed": 3,
        "max_train_steps": 100,
        "eval_every": 10,
        "eval_episodes": 5,
        "eval_metric": "episode_reward",
        "action_repeat": 1,
    }
    values.update(updates)
    return values


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_registry_separates_manifest_legacy_and_incomplete_runs(tmp_path):
    runs = tmp_path / "runs"
    complete = runs / "complete"
    legacy = runs / "legacy"
    incomplete = runs / "incomplete"

    config = _config()
    _write_json(complete / "config.json", config)
    _write_json(
        complete / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": "run-1",
            "lifecycle": {"status": "completed"},
            "source": {"commit": "a" * 40, "dirty": False},
            "config": {"sha256": "b" * 64},
            "progress": {"train_step": 100},
            "evaluation": {
                "metric": "episode_reward",
                "best_score": 500.0,
                "best_step": 90,
            },
            "outcome": {"stop_reason": "max_train_steps"},
        },
    )
    checkpoint = complete / "checkpoints" / "checkpoint_final.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    _write_json(legacy / "config.json", _config(seed=4))
    legacy_checkpoint = legacy / "checkpoints" / "checkpoint_final.pt"
    legacy_checkpoint.parent.mkdir()
    legacy_checkpoint.write_bytes(b"legacy")
    _write_json(incomplete / "config.json", _config(seed=5))

    csv_path = tmp_path / "reports" / "runs.csv"
    markdown_path = tmp_path / "reports" / "runs.md"
    records = generate_registry([runs], tmp_path, csv_path, markdown_path)

    by_path = {record.path: record for record in records}
    manifest_record = by_path["runs/complete"]
    assert manifest_record.disposition == "keep"
    assert manifest_record.comparable is True
    assert len(manifest_record.comparison_key) == 16
    assert manifest_record.checkpoint_bytes == len(b"checkpoint")

    legacy_record = by_path["runs/legacy"]
    assert legacy_record.lifecycle_status == "legacy-complete"
    assert legacy_record.disposition == "review"
    assert legacy_record.comparable is False
    assert "no versioned manifest" in legacy_record.comparability_reason

    assert by_path["runs/incomplete"].disposition == "incomplete"
    with csv_path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    report = markdown_path.read_text()
    assert "Indexed runs: 3" in report
    assert "No checkpoint contents were loaded" in report


def test_registry_marks_dirty_manifest_incomparable(tmp_path):
    run = tmp_path / "runs" / "dirty"
    _write_json(run / "config.json", _config())
    _write_json(
        run / "run_manifest.json",
        {
            "run_id": "dirty",
            "lifecycle": {"status": "completed"},
            "source": {"commit": "c" * 40, "dirty": True},
            "config": {"sha256": "d" * 64},
            "evaluation": {"metric": "episode_reward"},
        },
    )

    records = generate_registry(
        [tmp_path / "runs"],
        tmp_path,
        tmp_path / "runs.csv",
        tmp_path / "runs.md",
    )

    assert records[0].comparable is False
    assert "source tree was dirty" in records[0].comparability_reason
