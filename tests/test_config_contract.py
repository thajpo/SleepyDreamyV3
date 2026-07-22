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
    assert runtime_config.rssm_core == "reference"
    assert runtime_config.continue_head_layers == 1
    assert runtime_config.replay_sequence_mode == "stream"
    assert runtime_config.critic_slow_target is False
    assert runtime_config.optimizer_contract == "reference"
    assert runtime_config.optimizer_warmup_steps == 1000
    assert runtime_config.wm_lr == runtime_config.actor_lr
    assert runtime_config.actor_lr == runtime_config.critic_lr
    validate_config(runtime_config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(Config(), replay_burn_in=64), "burn-in < sequence_length"),
        (replace(Config(), checkpoint_interval=0), "checkpoint_interval"),
        (replace(Config(), d_hidden=63), "divisible by 16"),
        (replace(Config(), rssm_core="mystery"), "rssm_core"),
        (replace(Config(), continue_head_layers=2), "continue_head_layers"),
        (replace(Config(), optimizer_contract="mystery"), "optimizer_contract"),
        (
            replace(Config(), optimizer_warmup_steps=-1),
            "optimizer_warmup_steps",
        ),
        (
            replace(
                Config(),
                optimizer_contract="reference",
                actor_lr=3e-5,
            ),
            "requires equal",
        ),
        (
            replace(Config(), replay_sequence_mode="mystery"),
            "replay_sequence_mode",
        ),
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
    snapshot.pop("rssm_core")
    snapshot.pop("continue_head_layers")
    snapshot.pop("critic_slow_target")
    snapshot.pop("replay_sequence_mode")
    snapshot.pop("optimizer_contract")
    snapshot.pop("optimizer_warmup_steps")
    (run_dir / "config.json").write_text(json.dumps(snapshot))

    resumed = resolve_resume_config(
        replace(
            Config(),
            rssm_core="reference",
            continue_head_layers=1,
            critic_slow_target=False,
            replay_sequence_mode="stream",
        ),
        checkpoint_path,
        checkpoint={"world_model": {"continue_predictor.weight": object()}},
    )

    assert resumed.rssm_core == "legacy"
    assert resumed.continue_head_layers == 0
    assert resumed.critic_slow_target is True
    assert resumed.replay_sequence_mode == "episode"
    assert resumed.optimizer_contract == "legacy"
    assert resumed.optimizer_warmup_steps == 0


def test_resume_requires_explicit_semantic_migration(tmp_path):
    current = replace(Config(), continue_head_layers=1, critic_slow_target=False)
    resumed = resolve_resume_config(
        current,
        tmp_path / "checkpoint.pt",
        checkpoint={"world_model": {"continue_predictor.weight": object()}},
        allow_semantic_migration=True,
    )

    assert resumed is current


def test_resume_restores_reference_optimizer_contract_and_rates(tmp_path):
    checkpoint_config = replace(
        Config(),
        optimizer_contract="reference",
        optimizer_warmup_steps=1000,
        wm_lr=4e-5,
        actor_lr=4e-5,
        critic_lr=4e-5,
    )
    current = replace(
        Config(),
        wm_lr=3e-4,
        actor_lr=3e-5,
        critic_lr=8e-5,
    )

    resumed = resolve_resume_config(
        current,
        tmp_path / "checkpoint.pt",
        checkpoint={
            "config_snapshot": asdict(checkpoint_config),
            "world_model": {},
        },
    )

    assert resumed.optimizer_contract == "reference"
    assert resumed.optimizer_warmup_steps == 1000
    assert (resumed.wm_lr, resumed.actor_lr, resumed.critic_lr) == (
        4e-5,
        4e-5,
        4e-5,
    )


def test_resume_restores_checkpoint_authored_loss_and_bin_semantics(tmp_path):
    checkpoint_config = replace(
        Config(),
        normalize_advantages=True,
        free_bits_straight_through=True,
        b_start=-5,
        b_end=6,
        num_bins=127,
    )
    current = replace(
        Config(),
        normalize_advantages=False,
        free_bits_straight_through=False,
        b_start=-20,
        b_end=20,
        num_bins=255,
    )

    resumed = resolve_resume_config(
        current,
        tmp_path / "checkpoint.pt",
        checkpoint={
            "config_snapshot": asdict(checkpoint_config),
            "world_model": {},
        },
    )

    assert resumed.normalize_advantages is True
    assert resumed.free_bits_straight_through is True
    assert (resumed.b_start, resumed.b_end, resumed.num_bins) == (-5, 6, 127)


def test_resume_infers_reference_rssm_core_without_config_snapshot(tmp_path):
    resumed = resolve_resume_config(
        Config(),
        tmp_path / "checkpoint.pt",
        checkpoint={
            "world_model": {
                "dynin_deter.0.weight": object(),
                "continue_predictor.0.weight": object(),
            }
        },
    )

    assert resumed.rssm_core == "reference"
    assert resumed.continue_head_layers == 1


def test_trainer_requires_collector_queue_before_model_initialization(tmp_path):
    with pytest.raises(ValueError, match="requires a collector data queue"):
        WorldModelTrainer(
            Config(),
            data_queue=None,
            model_queues=None,
            log_dir=tmp_path,
        )
