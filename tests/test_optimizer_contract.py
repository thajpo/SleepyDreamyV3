from types import SimpleNamespace

import pytest
import torch

from dreamer.config import Config
from dreamer.models import (
    LaProp,
    adaptive_gradient_clipping,
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    symexp_twohot_bins,
)
from dreamer.runtime.replay_buffer import EnvData
from dreamer.trainer.core import WorldModelTrainer
from dreamer.trainer.forward import dreamer_step, replay_value_features
from dreamer.trainer.logging import create_step_metrics


def _optimizer(parameter, lr=4e-5):
    return LaProp([parameter], lr=lr)


@pytest.mark.parametrize(
    ("step", "expected_scale"),
    [(0, 0.0), (500, 0.5), (1000, 1.0), (1500, 1.0)],
)
def test_reference_warmup_applies_one_scale_to_every_optimizer(
    step, expected_scale
):
    trainer = object.__new__(WorldModelTrainer)
    trainer.train_step = step
    trainer.config = SimpleNamespace(
        optimizer_warmup_steps=1000,
        lr_cosine_decay=False,
        lr_cosine_min_factor=0.1,
        max_train_steps=3500,
        wm_lr=4e-5,
        actor_lr=4e-5,
        critic_lr=4e-5,
    )
    parameters = [torch.nn.Parameter(torch.ones(())) for _ in range(3)]
    trainer.wm_optimizer = _optimizer(parameters[0])
    trainer.actor_optimizer = _optimizer(parameters[1])
    trainer.critic_optimizer = _optimizer(parameters[2])

    trainer.apply_lr_schedule()

    for optimizer in (
        trainer.wm_optimizer,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
    ):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            4e-5 * expected_scale
        )


def test_replay_value_representation_gradient_is_contract_gated():
    source = torch.tensor([1.0, -2.0], requires_grad=True)
    reference = replay_value_features(
        source, optimizer_contract="reference", critic_replay_scale=0.3
    )
    assert reference.requires_grad
    reference.square().sum().backward()
    torch.testing.assert_close(source.grad, 2.0 * source.detach())

    source.grad = None
    legacy = replay_value_features(
        source, optimizer_contract="legacy", critic_replay_scale=0.3
    )
    assert not legacy.requires_grad
    assert source.grad is None


@pytest.mark.parametrize(
    ("optimizer_contract", "expected_representation_gradient"),
    [("reference", True), ("legacy", False)],
)
def test_replay_value_loss_obeys_observed_representation_contract(
    optimizer_contract, expected_representation_gradient
):
    torch.manual_seed(11)
    config = Config(
        batch_size=1,
        sequence_length=3,
        replay_burn_in=0,
        d_hidden=32,
        num_latents=4,
        rnn_n_blocks=1,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        num_dream_steps=2,
        rssm_core="reference",
        optimizer_contract=optimizer_contract,
        critic_replay_scale=0.3,
    )
    encoder, world_model = initialize_world_model("cpu", config, batch_size=1)
    actor = initialize_actor("cpu", config)
    critic = initialize_critic("cpu", config)
    critic_ema = initialize_critic("cpu", config)
    with torch.no_grad():
        critic.mlp[-1].weight.normal_(mean=0.0, std=0.1)
    critic_ema.load_state_dict(critic.state_dict())
    states = torch.randn(1, 3, 4)
    batch = EnvData(
        states=states,
        actions=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
        rewards=torch.tensor([[0.0, 1.0, 1.0]]),
        is_first=torch.tensor([[True, False, False]]),
        is_last=torch.tensor([[False, False, True]]),
        is_terminal=torch.tensor([[False, False, True]]),
        future_returns=None,
        continue_weights=torch.ones(1, 3),
        mask=torch.ones(1, 3),
    )
    all_tokens = encoder(states.reshape(3, 4)).reshape(1, 3, -1)
    world_model.init_state(1)

    result = dreamer_step(
        encoder=encoder,
        world_model=world_model,
        actor=actor,
        critic=critic,
        critic_ema=critic_ema,
        batch=batch,
        metrics=create_step_metrics(torch.device("cpu"), False),
        all_tokens=all_tokens,
        B=1,
        T=3,
        train_start_t=0,
        skip_actor=True,
        skip_critic=False,
        bins=symexp_twohot_bins(-20, 20, config.num_bins),
        return_scale=1.0,
        config=config,
        device=torch.device("cpu"),
        use_pixels=False,
        do_log_images=False,
    )
    result.total_critic_loss.backward()

    encoder_has_gradient = any(
        parameter.grad is not None and parameter.grad.norm().item() > 0.0
        for parameter in encoder.parameters()
    )
    recurrent_has_gradient = any(
        parameter.grad is not None and parameter.grad.norm().item() > 0.0
        for name, parameter in world_model.named_parameters()
        if name.startswith(
            (
                "dynin_deter.",
                "z_embedding.",
                "dynin_action.",
                "dynhid.",
                "dynhid_norm.",
                "dyngru.",
            )
        )
    )
    assert encoder_has_gradient is expected_representation_gradient
    assert recurrent_has_gradient is expected_representation_gradient


def test_equal_rate_split_laprop_matches_one_joint_optimizer():
    initial = [torch.tensor([0.5, -1.0]), torch.tensor([1.5]), torch.tensor([-0.2])]
    split = [torch.nn.Parameter(value.clone()) for value in initial]
    joint = [torch.nn.Parameter(value.clone()) for value in initial]
    split_optimizers = [_optimizer(parameter) for parameter in split]
    joint_optimizer = LaProp(joint, lr=4e-5)

    for _ in range(3):
        for optimizer in split_optimizers:
            optimizer.zero_grad()
        joint_optimizer.zero_grad()

        split_loss = (
            (split[0] * split[1]).sum()
            + split[1].square().sum()
            + split[2].square().sum()
        )
        joint_loss = (
            (joint[0] * joint[1]).sum()
            + joint[1].square().sum()
            + joint[2].square().sum()
        )
        split_loss.backward()
        joint_loss.backward()

        for parameter in split:
            adaptive_gradient_clipping([parameter], clip_factor=0.3)
        adaptive_gradient_clipping(joint, clip_factor=0.3)
        for optimizer in split_optimizers:
            optimizer.step()
        joint_optimizer.step()

    for split_parameter, joint_parameter in zip(split, joint):
        torch.testing.assert_close(split_parameter, joint_parameter)
