import json
from pathlib import Path
import shutil
import subprocess

import torch


def _run_training(output_dir: Path, checkpoint: Path | None = None):
    executable = shutil.which("dreamer-train")
    assert executable is not None, "dreamer-train must be installed for integration tests"
    max_steps = 2 if checkpoint else 1
    command = [
        executable,
        f"hydra.run.dir={output_dir}",
        "general.device=cpu",
        f"train.max_train_steps={max_steps}",
        "train.min_buffer_episodes=2",
        "train.batch_size=2",
        "train.sequence_length=4",
        "train.replay_burn_in=1",
        "train.eval_every=0",
        "train.checkpoint_interval=1",
    ]
    if checkpoint:
        command.append(f"checkpoint_path={checkpoint}")
    return subprocess.run(
        command,
        cwd=output_dir.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_installed_cli_checkpoint_resume_contract(tmp_path):
    first_dir = tmp_path / "first"
    resumed_dir = tmp_path / "resumed"

    first = _run_training(first_dir)
    assert first.returncode == 0, first.stdout + first.stderr
    first_checkpoint = first_dir / "checkpoints" / "checkpoint_final.pt"
    assert first_checkpoint.exists()

    resumed = _run_training(resumed_dir, checkpoint=first_checkpoint)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr

    first_manifest = json.loads((first_dir / "run_manifest.json").read_text())
    resumed_manifest = json.loads((resumed_dir / "run_manifest.json").read_text())
    resumed_checkpoint = torch.load(
        resumed_dir / "checkpoints" / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert first_manifest["lifecycle"]["status"] == "completed"
    assert first_manifest["progress"]["train_step"] == 1
    assert resumed_manifest["lifecycle"]["status"] == "completed"
    assert resumed_manifest["progress"]["train_step"] == 2
    assert resumed_manifest["tracking"]["resumed_from"] == str(first_checkpoint)
    assert resumed_checkpoint["step"] == 2
    assert resumed_checkpoint["run_id"] == resumed_manifest["run_id"]
