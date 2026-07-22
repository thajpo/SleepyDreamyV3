import json
from dataclasses import asdict, replace

import pytest
from hydra import compose, initialize_config_module

from dreamer.config import Config, ConfigValidationError, validate_config
from dreamer.main import dictconfig_to_config, resolve_resume_config, run_training
from dreamer.trainer.core import WorldModelTrainer


def test_hydra_yaml_defines_every_runtime_field():
    with initialize_config_module(config_module="dreamer.conf", version_base=None):
        hydra_config = compose(config_name="config")

    runtime_config = dictconfig_to_config(hydra_config)

    assert runtime_config.environment_name == "CartPole-v1"
    assert runtime_config.log_profile == "lean"
    assert runtime_config.num_bins == 255
    assert runtime_config.b_start == -20
    assert runtime_config.b_end == 20
    assert runtime_config.continue_head_layers == 1
    assert runtime_config.critic_slow_target is False
    validate_config(runtime_config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(Config(), replay_burn_in=64), "burn-in < sequence_length"),
        (replace(Config(), checkpoint_interval=0), "checkpoint_interval"),
        (replace(Config(), d_hidden=63), "divisible by 16"),
        (replace(Config(), continue_head_layers=2), "continue_head_layers"),
        (replace(Config(), actor_loss_mode="mystery"), "actor_loss_mode"),
        (
            replace(Config(), gamma=0.95, horizon=333, contdisc=True),
            "contdisc requires gamma",
        ),
        (
            replace(Config(), min_buffer_episodes=501),
            "min_buffer_episodes cannot exceed replay_buffer_size",
        ),
        (
            replace(Config(), use_pixels=False, n_observations=0),
            "state-only runs require n_observations",
        ),
    ],
)
def test_invalid_configs_fail_with_actionable_messages(config, message):
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(config)


def test_invalid_run_fails_before_creating_output(monkeypatch):
    def unexpected_output(*args, **kwargs):
        raise AssertionError("output directory should not be created")

    monkeypatch.setattr("dreamer.main.tempfile.mkdtemp", unexpected_output)

    with pytest.raises(ConfigValidationError, match="checkpoint_interval"):
        run_training(replace(Config(), dry_run=True, checkpoint_interval=0))


def test_resume_inherits_historical_checkpoint_semantics(tmp_path):
    run_dir = tmp_path / "historical"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    snapshot = asdict(Config())
    snapshot.pop("continue_head_layers")
    snapshot.pop("critic_slow_target")
    (run_dir / "config.json").write_text(json.dumps(snapshot))

    resumed = resolve_resume_config(
        replace(Config(), continue_head_layers=1, critic_slow_target=False),
        checkpoint_path,
        checkpoint={"world_model": {"continue_predictor.weight": object()}},
    )

    assert resumed.continue_head_layers == 0
    assert resumed.critic_slow_target is True


def test_resume_requires_explicit_semantic_migration(tmp_path):
    current = replace(Config(), continue_head_layers=1, critic_slow_target=False)
    resumed = resolve_resume_config(
        current,
        tmp_path / "checkpoint.pt",
        checkpoint={"world_model": {"continue_predictor.weight": object()}},
        allow_semantic_migration=True,
    )

    assert resumed is current


def test_trainer_requires_collector_queue_before_model_initialization(tmp_path):
    with pytest.raises(ValueError, match="requires a collector data queue"):
        WorldModelTrainer(
            Config(),
            data_queue=None,
            model_queues=None,
            log_dir=tmp_path,
        )
