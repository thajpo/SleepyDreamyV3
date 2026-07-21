import json
from pathlib import Path
import shutil
import subprocess

import torch


def _run_training(
    output_dir: Path,
    checkpoint: Path | None = None,
    extra_overrides: list[str] | None = None,
):
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
    if extra_overrides:
        command.extend(extra_overrides)
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


def test_cli_resume_restores_checkpoint_model_and_target_semantics(tmp_path):
    original_dir = tmp_path / "historical_semantics"
    resumed_dir = tmp_path / "resumed_semantics"
    original = _run_training(
        original_dir,
        extra_overrides=[
            "models.continue_head_layers=0",
            "train.critic_slow_target=true",
        ],
    )
    assert original.returncode == 0, original.stdout + original.stderr

    original_checkpoint = original_dir / "checkpoints" / "checkpoint_final.pt"
    resumed = _run_training(resumed_dir, checkpoint=original_checkpoint)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr

    resumed_config = json.loads((resumed_dir / "config.json").read_text())
    resumed_checkpoint = torch.load(
        resumed_dir / "checkpoints" / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert resumed_config["continue_head_layers"] == 0
    assert resumed_config["critic_slow_target"] is True
    assert "continue_predictor.weight" in resumed_checkpoint["world_model"]
    assert resumed_checkpoint["config_snapshot"]["continue_head_layers"] == 0
    assert resumed_checkpoint["config_snapshot"]["critic_slow_target"] is True


def test_initial_model_update_is_published_to_each_collector(tmp_path):
    result = _run_training(
        tmp_path / "multi_collector",
        extra_overrides=[
            "general.dry_run=true",
            "train.num_collectors=2",
            "train.replay_ratio=1000",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    # A one-update run may finish before both newly spawned collectors consume
    # their mailboxes. Publication is the deterministic integration boundary;
    # tests/test_model_broadcast.py verifies the contents of every mailbox.
    assert "model_weights_published version=0 delivered=[0, 1] pending=[]" in output
    assert "model_weights_published version=1" not in output


def test_actor_and_critic_train_from_first_update(tmp_path):
    output_dir = tmp_path / "joint_training"
    result = _run_training(output_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(
        output_dir / "checkpoints" / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["actor_optimizer"]["state"] != {}
    assert checkpoint["critic_optimizer"]["state"] != {}


def test_wm_ac_ratio_still_skips_actor_and_critic_together(tmp_path):
    output_dir = tmp_path / "wm_only_ratio_step"
    result = _run_training(
        output_dir,
        extra_overrides=["train.wm_ac_ratio=2"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(
        output_dir / "checkpoints" / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["actor_optimizer"]["state"] == {}
    assert checkpoint["critic_optimizer"]["state"] == {}
    assert checkpoint["wm_optimizer"]["state"] != {}
