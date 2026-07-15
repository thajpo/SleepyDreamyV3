"""Versioned, atomic metadata for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping
from uuid import uuid4

import torch


MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_git_state(repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture the full revision and whether source files were modified."""
    root = Path(repo_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current") or None
    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass instance or mapping")


def hash_config(config: Any) -> str:
    """Hash the normalized runtime configuration."""
    payload = json.dumps(
        _config_dict(config), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(log_dir: str | Path) -> Path:
    return Path(log_dir) / MANIFEST_FILENAME


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_run_manifest(log_dir: str | Path) -> dict[str, Any]:
    """Read a run manifest from its output directory."""
    return json.loads(manifest_path(log_dir).read_text())


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def update_run_manifest(log_dir: str | Path, updates: Mapping[str, Any]) -> None:
    """Merge fields into an existing manifest and replace it atomically."""
    current = read_run_manifest(log_dir)
    atomic_write_json(manifest_path(log_dir), _deep_merge(current, updates))


def create_run_manifest(
    log_dir: str | Path,
    config: Any,
    device: str,
    mlflow_run_id: str | None,
    checkpoint_path: str | None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Create the initial manifest before training subprocesses start."""
    started_at = utc_now()
    config_values = _config_dict(config)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": uuid4().hex,
        "lifecycle": {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "duration_seconds": None,
        },
        "source": capture_git_state(repo_root),
        "config": {
            "path": "config.json",
            "sha256": hash_config(config),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "device": device,
        },
        "tracking": {
            "mlflow_run_id": mlflow_run_id,
            "resumed_from": checkpoint_path,
        },
        "progress": {"train_step": 0, "env_steps": 0},
        "evaluation": {
            "metric": config_values.get("eval_metric", "episode_reward"),
            "latest_score": None,
            "latest_step": None,
            "best_score": None,
            "best_step": None,
        },
        "outcome": {"stop_reason": None, "error": None},
        "artifacts": {"best_checkpoint": None, "final_checkpoint": None},
    }
    atomic_write_json(manifest_path(log_dir), payload)
    return payload


def finish_run_manifest(
    log_dir: str | Path,
    status: str,
    error: str | None = None,
) -> None:
    """Record the parent-observed terminal status of a run."""
    current = read_run_manifest(log_dir)
    finished_at = datetime.now(timezone.utc)
    started_at = datetime.fromisoformat(current["lifecycle"]["started_at"])
    update_run_manifest(
        log_dir,
        {
            "lifecycle": {
                "status": status,
                "finished_at": finished_at.isoformat(),
                "duration_seconds": (finished_at - started_at).total_seconds(),
            },
            "outcome": {"error": error},
        },
    )
