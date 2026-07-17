"""Non-destructive indexing for training-run evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class RunRecord:
    path: str
    provenance: str
    lifecycle_status: str
    disposition: str
    comparable: bool
    comparability_reason: str
    comparison_key: str
    environment: str
    observation_mode: str
    seed: int | str
    budget_train_steps: int | str
    eval_every: int | str
    eval_episodes: int | str
    eval_metric: str
    git_commit: str
    git_dirty: bool | str
    config_sha256: str
    manifest_run_id: str
    train_step: int | str
    best_score: float | str
    best_step: int | str
    stop_reason: str
    checkpoint_count: int
    checkpoint_bytes: int
    run_bytes: int
    frame_multiplier: int | str
    frame_semantics: str


def _normalized_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _observation_mode(config: Mapping[str, Any]) -> str:
    pixels = bool(config.get("use_pixels", False))
    vector = int(config.get("n_observations", 0) or 0) > 0
    if pixels and vector:
        return "hybrid"
    if pixels:
        return "vision"
    return "vector"


def _frame_contract(config: Mapping[str, Any]) -> tuple[int | str, str]:
    environment = str(config.get("environment_name", ""))
    if environment.startswith("ALE/") and bool(config.get("atari_compat_mode")):
        multiplier = int(config.get("atari_frame_skip", 1) or 1)
        return multiplier, "ALE frames per recorded agent step"
    multiplier = int(config.get("action_repeat", 1) or 1)
    return multiplier, "environment steps per recorded action"


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            total += candidate.stat().st_size
        except OSError:
            continue
    return total


def discover_run_dirs(roots: Iterable[Path]) -> list[Path]:
    """Find run directories without treating MLflow internals as standalone runs."""
    discovered: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for filename in ("run_manifest.json", "config.json"):
            for candidate in root.rglob(filename):
                relative_parts = candidate.relative_to(root).parts
                if "mlruns" in relative_parts or ".hydra" in relative_parts:
                    continue
                discovered.add(candidate.parent.resolve())
    return sorted(discovered)


def index_run(run_dir: Path, repo_root: Path) -> RunRecord:
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "run_manifest.json"
    config: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    errors: list[str] = []

    if config_path.exists():
        try:
            config = _read_json(config_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json: {exc}")
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid run_manifest.json: {exc}")

    provenance = "manifest" if manifest else "legacy"
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints)
    has_final = any(path.name == "checkpoint_final.pt" for path in checkpoints)

    if errors:
        lifecycle_status = "invalid"
        disposition = "review"
    elif manifest:
        lifecycle_status = str(_nested(manifest, "lifecycle", "status", default="unknown"))
        disposition = "keep" if lifecycle_status == "completed" else "review"
    elif has_final:
        lifecycle_status = "legacy-complete"
        disposition = "review"
    elif checkpoints:
        lifecycle_status = "legacy-partial"
        disposition = "review"
    else:
        lifecycle_status = "incomplete"
        disposition = "incomplete"

    source_commit = str(_nested(manifest, "source", "commit", default="") or "")
    source_dirty = _nested(manifest, "source", "dirty", default="")
    config_sha = str(
        _nested(manifest, "config", "sha256", default="")
        or (_normalized_hash(config) if config else "")
    )
    environment = str(config.get("environment_name", ""))
    eval_metric = str(
        _nested(manifest, "evaluation", "metric", default="")
        or config.get("eval_metric", "episode_reward")
    )
    frame_multiplier, frame_semantics = _frame_contract(config)

    comparability_failures = list(errors)
    if provenance != "manifest":
        comparability_failures.append("legacy run has no versioned manifest")
    if lifecycle_status != "completed":
        comparability_failures.append("run did not record completed lifecycle")
    if not source_commit:
        comparability_failures.append("missing Git commit")
    if source_dirty is not False:
        comparability_failures.append("source tree was dirty or dirty state is unknown")
    if not config_sha:
        comparability_failures.append("missing config hash")
    required_protocol = {
        "environment": environment,
        "budget_train_steps": config.get("max_train_steps", ""),
        "eval_every": config.get("eval_every", ""),
        "eval_episodes": config.get("eval_episodes", ""),
        "eval_metric": eval_metric,
    }
    if any(value == "" for value in required_protocol.values()):
        comparability_failures.append("missing evaluation protocol fields")

    comparable = not comparability_failures
    comparison_key = ""
    if comparable:
        comparison_key = _normalized_hash(
            {
                **required_protocol,
                "observation_mode": _observation_mode(config),
                "git_commit": source_commit,
                "config_sha256": config_sha,
                "frame_multiplier": frame_multiplier,
            }
        )[:16]

    try:
        display_path = str(run_dir.relative_to(repo_root))
    except ValueError:
        display_path = str(run_dir)

    return RunRecord(
        path=display_path,
        provenance=provenance,
        lifecycle_status=lifecycle_status,
        disposition=disposition,
        comparable=comparable,
        comparability_reason="; ".join(dict.fromkeys(comparability_failures)),
        comparison_key=comparison_key,
        environment=environment,
        observation_mode=_observation_mode(config),
        seed=config.get("seed", ""),
        budget_train_steps=config.get("max_train_steps", ""),
        eval_every=config.get("eval_every", ""),
        eval_episodes=config.get("eval_episodes", ""),
        eval_metric=eval_metric,
        git_commit=source_commit,
        git_dirty=source_dirty,
        config_sha256=config_sha,
        manifest_run_id=str(manifest.get("run_id", "")),
        train_step=_nested(manifest, "progress", "train_step", default=""),
        best_score=_nested(manifest, "evaluation", "best_score", default=""),
        best_step=_nested(manifest, "evaluation", "best_step", default=""),
        stop_reason=str(_nested(manifest, "outcome", "stop_reason", default="") or ""),
        checkpoint_count=len(checkpoints),
        checkpoint_bytes=checkpoint_bytes,
        run_bytes=_directory_size(run_dir),
        frame_multiplier=frame_multiplier,
        frame_semantics=frame_semantics,
    )


def build_registry(roots: Iterable[Path], repo_root: Path) -> list[RunRecord]:
    return [index_run(path, repo_root) for path in discover_run_dirs(roots)]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(records: Iterable[RunRecord], path: Path) -> None:
    output = io.StringIO()
    fieldnames = [field.name for field in fields(RunRecord)]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(asdict(record))
    _atomic_write(path, output.getvalue())


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def write_markdown(records: list[RunRecord], path: Path) -> None:
    dispositions = Counter(record.disposition for record in records)
    provenance = Counter(record.provenance for record in records)
    total_bytes = sum(record.run_bytes for record in records)
    comparable = sum(record.comparable for record in records)
    lines = [
        "# Training Run Registry",
        "",
        "Generated non-destructively from run manifests and config snapshots. "
        "No checkpoint contents were loaded.",
        "",
        f"- Indexed runs: {len(records)}",
        f"- Indexed size: {_format_bytes(total_bytes)}",
        f"- Manifest / legacy: {provenance['manifest']} / {provenance['legacy']}",
        f"- Exact-comparison eligible: {comparable}",
        f"- Dispositions: {dict(sorted(dispositions.items()))}",
        "",
        "A blank comparison key means the run lacks the provenance or protocol "
        "needed for an exact comparison. Legacy runs are retained for review.",
        "",
        "| Run | Status | Disposition | Comparable | Checkpoints | Size |",
        "|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| `{record.path}` | {record.lifecycle_status} | {record.disposition} "
            f"| {'yes' if record.comparable else 'no'} | {record.checkpoint_count} "
            f"| {_format_bytes(record.run_bytes)} |"
        )
    lines.append("")
    _atomic_write(path, "\n".join(lines))


def generate_registry(
    roots: Iterable[Path], repo_root: Path, csv_path: Path, markdown_path: Path
) -> list[RunRecord]:
    records = build_registry(roots, repo_root)
    write_csv(records, csv_path)
    write_markdown(records, markdown_path)
    return records
