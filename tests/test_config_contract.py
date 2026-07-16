from dataclasses import replace

import pytest
from hydra import compose, initialize_config_module

from dreamer.config import Config, ConfigValidationError, validate_config
from dreamer.main import dictconfig_to_config, run_training
from dreamer.trainer.core import WorldModelTrainer


def test_hydra_yaml_defines_every_runtime_field():
    with initialize_config_module(config_module="dreamer.conf", version_base=None):
        hydra_config = compose(config_name="config")

    runtime_config = dictconfig_to_config(hydra_config)

    assert runtime_config.environment_name == "CartPole-v1"
    assert runtime_config.log_profile == "lean"
    assert runtime_config.num_bins == 255
    validate_config(runtime_config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(Config(), replay_burn_in=64), "burn-in < sequence_length"),
        (replace(Config(), checkpoint_interval=0), "checkpoint_interval"),
        (replace(Config(), d_hidden=63), "divisible by 16"),
        (replace(Config(), actor_loss_mode="mystery"), "actor_loss_mode"),
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


def test_trainer_requires_collector_queue_before_model_initialization(tmp_path):
    with pytest.raises(ValueError, match="requires a collector data queue"):
        WorldModelTrainer(
            Config(),
            data_queue=None,
            model_queue=None,
            log_dir=tmp_path,
        )
