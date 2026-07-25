import json
import math
from pathlib import Path

import pytest
import torch

from dreamer.config import Config, ConfigValidationError, validate_config
from dreamer.main import dictconfig_to_config
from dreamer.models import (
    ReferenceFeatureMLP,
    ReferenceMLP,
    ReferenceRMSNorm,
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    symlog,
)
from hydra import compose, initialize_config_module


FIXTURE = Path(__file__).parent / "fixtures" / "dreamerv3_e3f0224_oracle.json"


def reference_config(**changes) -> Config:
    values = dict(
        architecture_contract="reference_v3_state",
        rssm_core="reference",
        continue_head_layers=1,
        vector_encoder_mode="reference",
        posterior_head_layers=1,
        rnn_n_blocks=8,
        d_hidden=64,
        num_latents=32,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
    )
    values.update(changes)
    return Config(**values)


def module_types(module: torch.nn.Sequential) -> list[type]:
    return [type(layer) for layer in module]


def test_cartpole_hydra_selects_complete_reference_state_contract() -> None:
    with initialize_config_module(config_module="dreamer.conf", version_base=None):
        runtime = dictconfig_to_config(compose(config_name="config"))

    assert runtime.architecture_contract == "reference_v3_state"
    assert runtime.rnn_n_blocks == 8
    validate_config(runtime)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"use_pixels": True}, "pixel observations"),
        ({"rssm_core": "legacy"}, "rssm_core"),
        ({"continue_head_layers": 0}, "continue_head_layers"),
        ({"vector_encoder_mode": "legacy"}, "vector_encoder_mode"),
        ({"posterior_head_layers": 0}, "posterior_head_layers"),
        ({"rnn_n_blocks": 4}, "rnn_n_blocks"),
    ],
)
def test_reference_state_contract_rejects_hybrid_architectures(change, message) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(reference_config(**change))


def test_reference_rms_norm_matches_independent_jax_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text())["rms_norm"]
    norm = ReferenceRMSNorm(len(fixture["scale"]), eps=fixture["eps"])
    with torch.no_grad():
        norm.weight.copy_(torch.tensor(fixture["scale"]))
        norm.bias.copy_(torch.tensor(fixture["shift"]))

    actual = norm(torch.tensor(fixture["input"]))
    torch.testing.assert_close(
        actual, torch.tensor(fixture["output"]), rtol=1e-6, atol=1e-6
    )
    assert "bias" in norm.state_dict()


def test_reference_state_architecture_matches_pinned_size1m_topology() -> None:
    cfg = reference_config()
    encoder, world_model = initialize_world_model("cpu", cfg, batch_size=2)
    actor = initialize_actor("cpu", cfg)
    critic = initialize_critic("cpu", cfg)

    assert isinstance(encoder.MLP, ReferenceFeatureMLP)
    assert encoder.MLP.symlog_input is True
    assert module_types(encoder.MLP.mlp) == [
        torch.nn.Linear,
        ReferenceRMSNorm,
        torch.nn.SiLU,
    ] * 3

    assert world_model.n_blocks * world_model.d_hidden == 512
    assert world_model.n_latents == 32
    assert world_model.n_classes == 4
    assert isinstance(world_model.dynin_deter[1], ReferenceRMSNorm)
    assert isinstance(world_model.dynhid_norm, ReferenceRMSNorm)
    assert isinstance(world_model.posterior_head[1], ReferenceRMSNorm)
    assert module_types(world_model.dynamics_predictor.layers) == [
        torch.nn.Linear,
        ReferenceRMSNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
        ReferenceRMSNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
    ]

    assert isinstance(world_model.reward_predictor, ReferenceMLP)
    assert isinstance(world_model.continue_predictor, ReferenceMLP)
    assert isinstance(world_model.decoder.MLP, ReferenceMLP)
    assert module_types(world_model.reward_predictor.mlp) == [
        torch.nn.Linear,
        ReferenceRMSNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
    ]
    assert len(world_model.decoder.MLP.mlp) == 10
    assert isinstance(actor, ReferenceMLP)
    assert isinstance(critic, ReferenceMLP)
    assert len(actor.mlp) == 10
    assert len(critic.mlp) == 10


def test_reference_vector_encoder_applies_symlog_before_hidden_stack() -> None:
    cfg = reference_config()
    encoder, _world_model = initialize_world_model("cpu", cfg, batch_size=2)
    observations = torch.tensor([[0.0, 1.0, -10.0, 100.0]])

    actual = encoder(observations)
    expected = encoder.MLP.mlp(symlog(observations))
    torch.testing.assert_close(actual, expected)


def test_reference_initialization_has_zero_biases_and_scaled_outputs() -> None:
    torch.manual_seed(5)
    cfg = reference_config()
    _encoder, world_model = initialize_world_model("cpu", cfg, batch_size=2)
    actor = initialize_actor("cpu", cfg)
    critic = initialize_critic("cpu", cfg)

    modules = [world_model, actor, critic]
    for module in modules:
        for name, parameter in module.named_parameters():
            if name.endswith("bias"):
                torch.testing.assert_close(parameter, torch.zeros_like(parameter))

    actor_output = actor.mlp[-1]
    assert isinstance(actor_output, torch.nn.Linear)
    actor_bound = 2 * 1.1368 * 0.01 / math.sqrt(actor_output.in_features)
    assert actor_output.weight.abs().max().item() <= actor_bound + 1e-8
    assert actor_output.weight.abs().sum().item() > 0

    critic_output = critic.mlp[-1]
    reward_output = world_model.reward_predictor.mlp[-1]
    torch.testing.assert_close(
        critic_output.weight, torch.zeros_like(critic_output.weight)
    )
    torch.testing.assert_close(
        reward_output.weight, torch.zeros_like(reward_output.weight)
    )
    assert actor.mlp[0].weight.abs().sum().item() > 0
