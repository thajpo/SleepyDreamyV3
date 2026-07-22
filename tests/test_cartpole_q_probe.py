import json
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamer.config import Config
from dreamer.models import (
    initialize_actor,
    initialize_critic,
    initialize_q_critic,
    initialize_world_model,
)
from dreamer.models.dreaming import (
    enumerate_first_action_values,
    estimate_policy_lambda_action_values,
)
from scripts.probe_cartpole_q import (
    action_preference,
    hybrid_state_score,
    load_checkpoint_models,
    one_step_outcome,
    rollout_score,
)


@pytest.mark.parametrize(
    ("use_slow_target", "expected_key"),
    [(False, "critic"), (True, "critic_ema")],
)
def test_checkpoint_loader_selects_authored_critic_target(
    tmp_path, use_slow_target, expected_key
):
    cfg = replace(
        Config(),
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=1,
        critic_slow_target=use_slow_target,
    )
    actor = initialize_actor("cpu", cfg)
    critic = initialize_critic("cpu", cfg)
    critic_ema = initialize_critic("cpu", cfg)
    q_critic = initialize_q_critic("cpu", cfg)
    encoder, world_model = initialize_world_model("cpu", cfg, batch_size=1)
    with torch.no_grad():
        critic.mlp[-1].bias.fill_(1.0)
        critic_ema.mlp[-1].bias.fill_(2.0)

    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg)))
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "critic_ema": critic_ema.state_dict(),
            "q_critic": q_critic.state_dict(),
            "encoder": encoder.state_dict(),
            "world_model": world_model.state_dict(),
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_models(checkpoint_path, "cpu")

    loaded_critic = loaded[2]
    assert loaded[7] == expected_key
    expected_bias = 1.0 if expected_key == "critic" else 2.0
    assert torch.all(loaded_critic.mlp[-1].bias == expected_bias)


@pytest.mark.parametrize(
    ("critic_source", "expected_key", "expected_bias"),
    [("online", "critic", 1.0), ("slow", "critic_ema", 2.0)],
)
def test_checkpoint_loader_can_override_critic_source(
    tmp_path, critic_source, expected_key, expected_bias
):
    cfg = replace(
        Config(),
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=1,
        critic_slow_target=False,
    )
    actor = initialize_actor("cpu", cfg)
    critic = initialize_critic("cpu", cfg)
    critic_ema = initialize_critic("cpu", cfg)
    q_critic = initialize_q_critic("cpu", cfg)
    encoder, world_model = initialize_world_model("cpu", cfg, batch_size=1)
    with torch.no_grad():
        critic.mlp[-1].bias.fill_(1.0)
        critic_ema.mlp[-1].bias.fill_(2.0)

    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg)))
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "critic_ema": critic_ema.state_dict(),
            "q_critic": q_critic.state_dict(),
            "encoder": encoder.state_dict(),
            "world_model": world_model.state_dict(),
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_models(
        checkpoint_path, "cpu", critic_source=critic_source
    )

    assert loaded[7] == expected_key
    assert torch.all(loaded[2].mlp[-1].bias == expected_bias)


def test_checkpoint_loader_rejects_unknown_critic_source(tmp_path):
    checkpoint_path = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="critic_source"):
        load_checkpoint_models(
            checkpoint_path, "cpu", critic_source="mystery"
        )


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (np.array([0.0, 0.0, 0.03, 0.0], dtype=np.float32), 0),
        (np.array([0.0, 0.0, -0.03, 0.0], dtype=np.float32), 1),
        (np.array([2.39, 1.0, 0.0, 0.0], dtype=np.float32), 1),
    ],
)
def test_perfect_one_step_prediction_preserves_real_rollout_score(state, action):
    env = gym.make("CartPole-v1")
    horizon = 10
    try:
        real_score = rollout_score(env, state, action, horizon)
        real_next_state, _done = one_step_outcome(env, state, action)
        hybrid_score = hybrid_state_score(env, real_next_state, horizon)
    finally:
        env.close()

    assert hybrid_score == real_score


def test_perfect_one_step_predictions_preserve_action_preference():
    env = gym.make("CartPole-v1")
    state = np.array([0.0, 0.1, 0.04, -0.1], dtype=np.float32)
    horizon = 30
    try:
        real_scores = [rollout_score(env, state, action, horizon) for action in (0, 1)]
        hybrid_scores = []
        for action in (0, 1):
            next_state, _done = one_step_outcome(env, state, action)
            hybrid_scores.append(hybrid_state_score(env, next_state, horizon))
    finally:
        env.close()

    assert action_preference(hybrid_scores) == action_preference(real_scores)


class _BootstrapProbeWorldModel:
    n_blocks = 1
    n_latents = 1
    n_classes = 2

    def step_dynamics(self, z_embed, action_onehot, h_state):
        del z_embed
        h_next = h_state + action_onehot[:, 1:2]
        prior_logits = torch.zeros(action_onehot.shape[0], 1, 2)
        return h_next, prior_logits

    def join_h_and_z(self, h_state, z_state):
        return torch.cat([h_state, z_state.reshape(z_state.shape[0], -1)], dim=-1)

    def z_embedding(self, z_state):
        return z_state

    def continue_predictor(self, h_z):
        return torch.full((h_z.shape[0], 1), 20.0)

    def reward_predictor(self, h_z):
        return torch.zeros(h_z.shape[0], 3)


class _BootstrapProbeCritic(torch.nn.Module):
    def forward(self, h_z):
        signal = 10.0 * h_z[..., :1]
        return torch.cat([-signal, torch.zeros_like(signal), signal], dim=-1)


def test_first_action_enumeration_can_disable_terminal_critic_bootstrap():
    world_model = _BootstrapProbeWorldModel()
    critic = _BootstrapProbeCritic()
    kwargs = {
        "initial_h_z": torch.zeros(1, 3),
        "initial_z_embed": torch.zeros(1, 2),
        "actor": None,
        "critic": critic,
        "world_model": world_model,
        "n_actions": 2,
        "d_hidden": 1,
        "bins": torch.tensor([-1.0, 0.0, 1.0]),
        "gamma": 1.0,
        "horizon": 1,
    }

    with_bootstrap = enumerate_first_action_values(**kwargs, bootstrap_value=True)
    without_bootstrap = enumerate_first_action_values(**kwargs, bootstrap_value=False)

    assert with_bootstrap[0, 1] > with_bootstrap[0, 0]
    assert torch.allclose(without_bootstrap, torch.zeros_like(without_bootstrap))


def test_first_action_enumeration_rejects_unknown_latent_mode():
    world_model = _BootstrapProbeWorldModel()
    with pytest.raises(ValueError, match="unsupported latent_mode"):
        enumerate_first_action_values(
            initial_h_z=torch.zeros(1, 3),
            initial_z_embed=torch.zeros(1, 2),
            actor=None,
            critic=_BootstrapProbeCritic(),
            world_model=world_model,
            n_actions=2,
            d_hidden=1,
            bins=torch.tensor([-1.0, 0.0, 1.0]),
            gamma=1.0,
            horizon=1,
            latent_mode="unknown",
        )


def test_policy_lambda_action_values_condition_on_first_action():
    world_model = _BootstrapProbeWorldModel()
    means, standard_errors = estimate_policy_lambda_action_values(
        initial_h_z=torch.zeros(1, 3),
        initial_z_embed=torch.zeros(1, 2),
        actor=None,
        critic=_BootstrapProbeCritic(),
        world_model=world_model,
        n_actions=2,
        d_hidden=1,
        bins=torch.tensor([-1.0, 0.0, 1.0]),
        gamma=1.0,
        lam=0.95,
        horizon=1,
        samples=8,
        generator=torch.Generator().manual_seed(7),
    )

    assert means.shape == (1, 2)
    assert standard_errors.shape == (1, 2)
    assert means[0, 1] > means[0, 0]
    assert torch.allclose(standard_errors, torch.zeros_like(standard_errors))


@pytest.mark.parametrize(("horizon", "samples"), [(0, 8), (1, 1)])
def test_policy_lambda_action_values_reject_invalid_budget(horizon, samples):
    with pytest.raises(ValueError):
        estimate_policy_lambda_action_values(
            initial_h_z=torch.zeros(1, 3),
            initial_z_embed=torch.zeros(1, 2),
            actor=None,
            critic=_BootstrapProbeCritic(),
            world_model=_BootstrapProbeWorldModel(),
            n_actions=2,
            d_hidden=1,
            bins=torch.tensor([-1.0, 0.0, 1.0]),
            gamma=1.0,
            lam=0.95,
            horizon=horizon,
            samples=samples,
        )
