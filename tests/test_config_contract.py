import json
from dataclasses import asdict, replace

import pytest
import torch
from hydra import compose, initialize_config_module

from dreamer.config import (
    Config,
    ConfigValidationError,
    load_checkpoint_config,
    validate_config,
)
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
    assert runtime_config.vector_encoder_mode == "reference"
    assert runtime_config.posterior_head_layers == 1
    assert runtime_config.replay_sequence_mode == "stream"
    assert runtime_config.replay_row_alignment == "reference"
    assert runtime_config.replay_burn_in == 20
    assert runtime_config.online_replay is True
    assert runtime_config.continuous_replay_delivery is True
    assert runtime_config.replay_evidence_samples == 0
    assert runtime_config.critic_slow_target is False
    assert runtime_config.critic_ema_target == "mean_twohot"
    assert runtime_config.optimizer_contract == "reference"
    assert runtime_config.laprop_bias_correction is True
    assert runtime_config.optimizer_warmup_steps == 1000
    assert runtime_config.actor_warmup_steps == 0
    assert runtime_config.actor_unimix == 0.01
    assert runtime_config.weight_imagination_starts is True
    assert runtime_config.state_loss_mode == "reference_sum"
    assert runtime_config.recent_fraction == 0.0
    assert runtime_config.wm_lr == runtime_config.actor_lr
    assert runtime_config.actor_lr == runtime_config.critic_lr
    validate_config(runtime_config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(Config(), replay_burn_in=64), "burn-in < sequence_length"),
        (replace(Config(), replay_row_alignment="mystery"), "replay_row_alignment"),
        (replace(Config(), checkpoint_interval=0), "checkpoint_interval"),
        (replace(Config(), d_hidden=63), "divisible by 16"),
        (replace(Config(), encoder_mlp_n_layers=0), "encoder_mlp_n_layers"),
        (replace(Config(), rssm_core="mystery"), "rssm_core"),
        (replace(Config(), continue_head_layers=2), "continue_head_layers"),
        (replace(Config(), vector_encoder_mode="mystery"), "vector_encoder_mode"),
        (replace(Config(), posterior_head_layers=2), "posterior_head_layers"),
        (replace(Config(), optimizer_contract="mystery"), "optimizer_contract"),
        (replace(Config(), critic_ema_target="mystery"), "critic_ema_target"),
        (replace(Config(), state_loss_mode="mystery"), "state_loss_mode"),
        (
            replace(Config(), optimizer_warmup_steps=-1),
            "optimizer_warmup_steps",
        ),
        (replace(Config(), actor_warmup_steps=-1), "actor_warmup_steps"),
        (replace(Config(), actor_unimix=-0.01), "actor_unimix"),
        (replace(Config(), actor_unimix=1.01), "actor_unimix"),
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
        (
            replace(Config(), online_replay=True, replay_sequence_mode="episode"),
            "online_replay requires",
        ),
        (
            replace(
                Config(),
                continuous_replay_delivery=True,
                replay_sequence_mode="episode",
            ),
            "continuous_replay_delivery requires",
        ),
        (
            replace(
                Config(),
                continuous_replay_delivery=True,
                replay_sequence_mode="stream",
                critic_real_return_scale=0.1,
            ),
            "incompatible with critic_real_return_scale",
        ),
        (
            replace(Config(), replay_evidence_samples=-1),
            "replay_evidence_samples must be >= 0",
        ),
        (
            replace(Config(), replay_evidence_samples=1),
            "replay_evidence_samples requires replay_sequence_mode",
        ),
        (
            replace(
                Config(),
                replay_evidence_samples=1,
                replay_sequence_mode="stream",
                n_observations=0,
                use_pixels=True,
            ),
            "replay_evidence_samples requires vector observations",
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
    snapshot.pop("vector_encoder_mode")
    snapshot.pop("posterior_head_layers")
    snapshot.pop("critic_slow_target")
    snapshot.pop("critic_ema_target")
    snapshot.pop("replay_sequence_mode")
    snapshot.pop("online_replay")
    snapshot.pop("continuous_replay_delivery")
    snapshot.pop("optimizer_contract")
    snapshot.pop("laprop_bias_correction")
    snapshot.pop("optimizer_warmup_steps")
    snapshot.pop("weight_imagination_starts")
    snapshot.pop("state_loss_mode")
    snapshot.pop("actor_unimix")
    snapshot["actor_warmup_steps"] = 3000
    snapshot["gamma"] = 0.9
    snapshot["horizon"] = 10
    snapshot["contdisc"] = True
    (run_dir / "config.json").write_text(json.dumps(snapshot))

    resumed = resolve_resume_config(
        replace(
            Config(),
            rssm_core="reference",
            continue_head_layers=1,
            vector_encoder_mode="reference",
            posterior_head_layers=1,
            critic_slow_target=False,
            replay_sequence_mode="stream",
            online_replay=True,
            laprop_bias_correction=True,
            actor_warmup_steps=0,
            gamma=0.75,
            horizon=4,
            contdisc=True,
        ),
        checkpoint_path,
        checkpoint={"world_model": {"continue_predictor.weight": object()}},
    )

    assert resumed.rssm_core == "legacy"
    assert resumed.continue_head_layers == 0
    assert resumed.vector_encoder_mode == "legacy"
    assert resumed.posterior_head_layers == 0
    assert resumed.critic_slow_target is True
    assert resumed.critic_ema_target == "distribution"
    assert resumed.replay_sequence_mode == "episode"
    assert resumed.online_replay is False
    assert resumed.continuous_replay_delivery is False
    assert resumed.optimizer_contract == "legacy"
    assert resumed.laprop_bias_correction is False
    assert resumed.optimizer_warmup_steps == 0
    assert resumed.actor_warmup_steps == 3000
    assert resumed.actor_unimix == 0.01
    assert (resumed.gamma, resumed.horizon, resumed.contdisc) == (0.9, 10, True)
    assert resumed.weight_imagination_starts is False
    assert resumed.state_loss_mode == "legacy_half_mean"


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
        laprop_bias_correction=True,
        optimizer_warmup_steps=1000,
        wm_lr=4e-5,
        actor_lr=4e-5,
        critic_lr=4e-5,
        actor_unimix=0.10,
        replay_sequence_mode="stream",
        online_replay=True,
        continuous_replay_delivery=True,
    )
    current = replace(
        Config(),
        wm_lr=3e-4,
        actor_lr=3e-5,
        critic_lr=8e-5,
        laprop_bias_correction=False,
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
    assert resumed.laprop_bias_correction is True
    assert resumed.optimizer_warmup_steps == 1000
    assert resumed.actor_unimix == 0.10
    assert resumed.replay_sequence_mode == "stream"
    assert resumed.online_replay is True
    assert resumed.continuous_replay_delivery is True
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
        weight_imagination_starts=True,
        critic_ema_target="mean_twohot",
        state_loss_mode="reference_sum",
    )
    current = replace(
        Config(),
        normalize_advantages=False,
        free_bits_straight_through=False,
        b_start=-20,
        b_end=20,
        num_bins=255,
        weight_imagination_starts=False,
        critic_ema_target="distribution",
        state_loss_mode="legacy_half_mean",
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
    assert resumed.weight_imagination_starts is True
    assert resumed.critic_ema_target == "mean_twohot"
    assert resumed.state_loss_mode == "reference_sum"


def test_resume_restores_all_objective_semantics_by_default(tmp_path):
    checkpoint_config = replace(
        Config(),
        max_train_steps=100,
        eval_metric="episode_length",
        lam=0.7,
        num_dream_steps=9,
        critic_ema_decay=0.95,
        critic_ema_regularizer=0.4,
        actor_entropy_coef=0.002,
        beta_dyn=0.8,
        beta_rep=0.2,
        beta_pred=1.3,
        critic_replay_scale=0.5,
        prior_state_pred_scale=0.25,
        replay_ratio=4.0,
        recent_fraction=0.2,
    )
    current = replace(
        checkpoint_config,
        max_train_steps=200,
        eval_metric="episode_reward",
        lam=0.9,
        num_dream_steps=15,
        critic_ema_decay=0.98,
        critic_ema_regularizer=1.0,
        actor_entropy_coef=3e-4,
        beta_dyn=1.0,
        beta_rep=0.1,
        beta_pred=1.0,
        critic_replay_scale=0.3,
        prior_state_pred_scale=0.0,
        replay_ratio=1.0,
        recent_fraction=0.0,
    )

    resumed = resolve_resume_config(
        current,
        tmp_path / "checkpoint.pt",
        checkpoint={
            "config_snapshot": asdict(checkpoint_config),
            "world_model": {},
        },
    )

    assert resumed.max_train_steps == 200
    assert resumed.eval_metric == "episode_length"
    assert resumed.lam == 0.7
    assert resumed.num_dream_steps == 9
    assert resumed.critic_ema_decay == 0.95
    assert resumed.critic_ema_regularizer == 0.4
    assert resumed.actor_entropy_coef == 0.002
    assert (resumed.beta_dyn, resumed.beta_rep, resumed.beta_pred) == (0.8, 0.2, 1.3)
    assert resumed.critic_replay_scale == 0.5
    assert resumed.prior_state_pred_scale == 0.25
    assert resumed.replay_ratio == 4.0
    assert resumed.recent_fraction == 0.2


def test_shifted_reference_checkpoint_gets_v1_compatibility_contract(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "config_snapshot": asdict(
            replace(Config(), architecture_contract="reference_v3_state")
        ),
        "encoder": {
            "MLP.mlp.0.weight": torch.zeros(2, 2),
            "MLP.mlp.1.weight": torch.ones(2),
            "MLP.mlp.1.bias": torch.zeros(2),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    loaded = load_checkpoint_config(checkpoint_path)

    assert loaded is not None
    assert loaded.architecture_contract == "reference_v3_state_v1"


def test_scale_only_reference_checkpoint_keeps_current_contract(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "config_snapshot": asdict(
            replace(Config(), architecture_contract="reference_v3_state")
        ),
        "encoder": {
            "MLP.mlp.0.weight": torch.zeros(2, 2),
            "MLP.mlp.1.weight": torch.ones(2),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    loaded = load_checkpoint_config(checkpoint_path)

    assert loaded is not None
    assert loaded.architecture_contract == "reference_v3_state"


def test_resume_infers_reference_rssm_core_without_config_snapshot(tmp_path):
    resumed = resolve_resume_config(
        Config(),
        tmp_path / "checkpoint.pt",
        checkpoint={
            "world_model": {
                "dynin_deter.0.weight": object(),
                "continue_predictor.0.weight": object(),
                "posterior_head.0.weight": object(),
            },
            "encoder": {"MLP.mlp.1.weight": object()},
        },
    )

    assert resumed.rssm_core == "reference"
    assert resumed.continue_head_layers == 1
    assert resumed.vector_encoder_mode == "reference"
    assert resumed.posterior_head_layers == 1
    assert resumed.online_replay is False
    assert resumed.continuous_replay_delivery is False
    assert resumed.laprop_bias_correction is False
    assert (resumed.gamma, resumed.horizon, resumed.contdisc) == (
        0.997,
        333,
        True,
    )


def test_unsnapshotted_resume_uses_historical_objective_defaults(tmp_path):
    resumed = resolve_resume_config(
        replace(
            Config(),
            lam=0.7,
            num_dream_steps=9,
            critic_ema_decay=0.95,
            critic_ema_regularizer=0.4,
            actor_entropy_coef=0.002,
            beta_dyn=0.8,
            beta_rep=0.2,
            beta_pred=1.3,
        ),
        tmp_path / "checkpoint.pt",
        checkpoint={
            "world_model": {
                "dynin_deter.0.weight": object(),
                "continue_predictor.0.weight": object(),
                "posterior_head.0.weight": object(),
            },
            "encoder": {"MLP.mlp.1.weight": object()},
        },
    )

    historical = Config()
    assert resumed.lam == historical.lam
    assert resumed.num_dream_steps == historical.num_dream_steps
    assert resumed.critic_ema_decay == historical.critic_ema_decay
    assert resumed.critic_ema_regularizer == historical.critic_ema_regularizer
    assert resumed.actor_entropy_coef == historical.actor_entropy_coef
    assert (resumed.beta_dyn, resumed.beta_rep, resumed.beta_pred) == (
        historical.beta_dyn,
        historical.beta_rep,
        historical.beta_pred,
    )


def test_resume_infers_legacy_observation_posterior_without_snapshot(tmp_path):
    resumed = resolve_resume_config(
        replace(
            Config(),
            vector_encoder_mode="reference",
            posterior_head_layers=1,
        ),
        tmp_path / "checkpoint.pt",
        checkpoint={
            "world_model": {
                "_W_ir": object(),
                "continue_predictor.weight": object(),
                "posterior_head.weight": object(),
            },
            "encoder": {"MLP.mlp.0.weight": object()},
        },
    )

    assert resumed.vector_encoder_mode == "legacy"
    assert resumed.posterior_head_layers == 0


def test_trainer_requires_collector_queue_before_model_initialization(tmp_path):
    with pytest.raises(ValueError, match="requires a collector data queue"):
        WorldModelTrainer(
            Config(),
            data_queue=None,
            model_queues=None,
            log_dir=tmp_path,
        )
