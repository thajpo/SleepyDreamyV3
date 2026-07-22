import json
from dataclasses import asdict

import torch

from dreamer.config import Config
from dreamer.inspect import infer_config_from_checkpoint


def test_inspector_preserves_actor_warmup_in_historical_config(tmp_path):
    run_dir = tmp_path / "historical_run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    data = asdict(Config())
    data["actor_warmup_steps"] = 3000
    (run_dir / "config.json").write_text(json.dumps(data))

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert config.actor_warmup_steps == 3000
    assert config.d_hidden == Config().d_hidden


def test_inspector_uses_linear_continuation_head_for_historical_config(tmp_path):
    run_dir = tmp_path / "historical_run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    data = asdict(Config())
    data.pop("continue_head_layers")
    (run_dir / "config.json").write_text(json.dumps(data))

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert config.continue_head_layers == 0


def test_inspector_uses_legacy_rssm_core_for_historical_config(tmp_path):
    run_dir = tmp_path / "historical_run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    data = asdict(Config())
    data.pop("rssm_core")
    (run_dir / "config.json").write_text(json.dumps(data))

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert config.rssm_core == "legacy"


def test_inspector_uses_legacy_observation_posterior_for_historical_config(tmp_path):
    run_dir = tmp_path / "historical_run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    data = asdict(Config())
    data.pop("vector_encoder_mode")
    data.pop("posterior_head_layers")
    (run_dir / "config.json").write_text(json.dumps(data))

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert config.vector_encoder_mode == "legacy"
    assert config.posterior_head_layers == 0


def test_inspector_uses_slow_critic_target_for_historical_config(tmp_path):
    run_dir = tmp_path / "historical_run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    data = asdict(Config())
    data.pop("critic_slow_target")
    (run_dir / "config.json").write_text(json.dumps(data))

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert config.critic_slow_target is True


def test_inspector_reads_config_from_portable_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint_final.pt"
    snapshot = asdict(Config(continue_head_layers=1, critic_slow_target=False))
    torch.save({"config_snapshot": snapshot}, checkpoint_path)

    config = infer_config_from_checkpoint(checkpoint_path, config_name=None)

    assert config.continue_head_layers == 1
    assert config.critic_slow_target is False
