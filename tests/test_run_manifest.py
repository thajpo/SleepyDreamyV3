import json
from pathlib import Path

from dreamer.config import Config
from dreamer.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    create_run_manifest,
    file_sha256,
    finish_run_manifest,
    hash_config,
    read_run_manifest,
    update_run_manifest,
)


def test_manifest_records_full_provenance_and_merges_updates(tmp_path):
    config = Config(seed=17)
    created = create_run_manifest(
        log_dir=tmp_path,
        config=config,
        device="cpu",
        mlflow_run_id="mlflow-1",
        checkpoint_path=None,
    )

    assert created["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert len(created["source"]["commit"]) == 40
    assert isinstance(created["source"]["dirty"], bool)
    assert created["config"]["sha256"] == hash_config(config)

    update_run_manifest(
        tmp_path,
        {
            "evaluation": {"best_score": 42.0, "best_step": 100},
            "progress": {"train_step": 100},
        },
    )
    finish_run_manifest(tmp_path, status="completed")
    manifest = read_run_manifest(tmp_path)

    assert manifest["evaluation"]["best_score"] == 42.0
    assert manifest["evaluation"]["latest_score"] is None
    assert manifest["progress"]["train_step"] == 100
    assert manifest["lifecycle"]["status"] == "completed"
    assert manifest["lifecycle"]["duration_seconds"] >= 0


def test_file_hash_matches_artifact_bytes(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"sleepy-dreamy")

    assert file_sha256(artifact) == (
        "6e54dbfc66b8a338cecc48feb7f3fa3fcd66e410f0ba9e215590d7bb2134e13c"
    )


def test_manifest_json_is_complete_after_atomic_write(tmp_path):
    create_run_manifest(tmp_path, Config(), "cpu", None, None)

    payload = json.loads((Path(tmp_path) / "run_manifest.json").read_text())
    assert payload["lifecycle"]["status"] == "running"
    assert not list(Path(tmp_path).glob("*.tmp"))
