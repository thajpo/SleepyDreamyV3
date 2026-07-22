from types import SimpleNamespace

import pytest
import torch

from dreamer.models.decoder import ObservationDecoder
from dreamer.models.losses import compute_actor_critic_losses, compute_wm_loss
from dreamer.models.math_utils import symlog


def test_compute_wm_loss_pixel_only_no_state_is_finite():
    batch_size = 4
    n_bins = 5
    n_latents = 8
    n_classes = 4

    obs_reconstruction = {"pixels": torch.randn(batch_size, 3, 64, 64)}
    obs_target = {"pixels": torch.randint(0, 256, (batch_size, 3, 64, 64)).float()}

    reward_logits = torch.randn(batch_size, n_bins)
    reward_target = torch.randn(batch_size)
    terminated = torch.zeros(batch_size, dtype=torch.bool)
    continue_logits = torch.randn(batch_size, 1)

    posterior_logits = torch.randn(batch_size, n_latents, n_classes)
    prior_logits = torch.randn(batch_size, n_latents, n_classes)

    bins = torch.linspace(-2.0, 2.0, n_bins)
    cfg = SimpleNamespace(beta_dyn=1.0, beta_rep=0.1, beta_pred=1.0)

    total_loss, loss_dict = compute_wm_loss(
        obs_reconstruction=obs_reconstruction,
        obs_t=obs_target,
        reward_dist=reward_logits,
        reward_t=reward_target,
        terminated_t=terminated,
        continue_logits=continue_logits,
        posterior_logits=posterior_logits,
        prior_logits=prior_logits,
        B=bins,
        config=cfg,
        device=reward_logits.device,
        use_pixels=True,
        sample_mask=torch.ones(batch_size),
    )

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_dict["prediction_pixel"])
    assert loss_dict["prediction_vector"].item() == 0.0


def _free_bits_gradient_case(straight_through):
    batch_size = 4
    n_bins = 5
    n_latents = 8
    n_classes = 4

    posterior_logits = torch.randn(batch_size, n_latents, n_classes)
    prior_logits = (posterior_logits + 0.01 * torch.randn_like(posterior_logits))
    prior_logits = prior_logits.detach().requires_grad_()

    cfg = SimpleNamespace(
        beta_dyn=1.0,
        beta_rep=0.0,
        beta_pred=0.0,
        free_bits_straight_through=straight_through,
    )
    total_loss, loss_dict = compute_wm_loss(
        obs_reconstruction={"state": torch.zeros(batch_size, 1)},
        obs_t={"state": torch.zeros(batch_size, 1)},
        reward_dist=torch.zeros(batch_size, n_bins),
        reward_t=torch.zeros(batch_size),
        terminated_t=torch.zeros(batch_size, dtype=torch.bool),
        continue_logits=torch.zeros(batch_size, 1),
        posterior_logits=posterior_logits,
        prior_logits=prior_logits,
        B=torch.linspace(-2.0, 2.0, n_bins),
        config=cfg,
        device=prior_logits.device,
        use_pixels=False,
        sample_mask=torch.ones(batch_size),
    )

    assert loss_dict["kl_dynamics_raw"].item() < 1.0
    assert loss_dict["dynamics"].item() == 1.0

    total_loss.backward()
    return prior_logits.grad


def test_free_bits_preserves_prior_gradient_below_threshold():
    grad = _free_bits_gradient_case(straight_through=True)
    assert grad is not None
    assert grad.norm().item() > 0.0


def test_hard_free_bits_clamp_blocks_prior_gradient_below_threshold():
    grad = _free_bits_gradient_case(straight_through=False)
    assert grad is not None
    assert grad.norm().item() == 0.0


def test_state_decoder_loss_uses_raw_target_symlog_once():
    batch_size = 3
    n_bins = 5
    n_latents = 8
    n_classes = 4
    raw_state = torch.tensor(
        [[0.0, 1.0, -2.0, 3.0], [0.5, -0.5, 2.0, -3.0], [1.5, 0.0, -1.0, 2.5]]
    )

    cfg = SimpleNamespace(beta_dyn=0.0, beta_rep=0.0, beta_pred=1.0)
    _total_loss, loss_dict = compute_wm_loss(
        obs_reconstruction={"state": symlog(raw_state)},
        obs_t={"state": raw_state},
        reward_dist=torch.zeros(batch_size, n_bins),
        reward_t=torch.zeros(batch_size),
        terminated_t=torch.zeros(batch_size, dtype=torch.bool),
        continue_logits=torch.zeros(batch_size, 1),
        posterior_logits=torch.zeros(batch_size, n_latents, n_classes),
        prior_logits=torch.zeros(batch_size, n_latents, n_classes),
        B=torch.linspace(-2.0, 2.0, n_bins),
        config=cfg,
        device=raw_state.device,
        use_pixels=False,
        sample_mask=torch.ones(batch_size),
    )

    assert loss_dict["prediction_vector"].item() == 0.0


def test_continue_importance_weights_only_change_continue_loss():
    batch_size = 2
    n_bins = 5
    n_latents = 2
    n_classes = 3
    common = dict(
        obs_reconstruction={"state": torch.zeros(batch_size, 1)},
        obs_t={"state": torch.zeros(batch_size, 1)},
        reward_dist=torch.zeros(batch_size, n_bins),
        reward_t=torch.zeros(batch_size),
        terminated_t=torch.tensor([True, False]),
        continue_logits=torch.zeros(batch_size, 1),
        posterior_logits=torch.zeros(batch_size, n_latents, n_classes),
        prior_logits=torch.zeros(batch_size, n_latents, n_classes),
        B=torch.linspace(-2.0, 2.0, n_bins),
        config=SimpleNamespace(
            beta_dyn=0.0,
            beta_rep=0.0,
            beta_pred=1.0,
            contdisc=False,
        ),
        device=torch.device("cpu"),
        use_pixels=False,
        sample_mask=torch.ones(batch_size),
    )

    _plain_total, plain = compute_wm_loss(**common)
    _weighted_total, weighted = compute_wm_loss(
        **common,
        continue_loss_weights=torch.tensor([3.0, 0.5]),
    )

    assert weighted["prediction_continue"].item() == pytest.approx(
        plain["prediction_continue"].item() * 1.75
    )
    for key in (
        "prediction_vector",
        "prediction_reward",
        "dynamics",
        "representation",
    ):
        assert weighted[key].item() == pytest.approx(plain[key].item())


def test_actor_advantage_normalization_centers_constant_returns():
    horizon = 2
    batch_size = 3
    n_bins = 5
    n_actions = 2
    bins = torch.linspace(-2.0, 2.0, n_bins)

    values = torch.zeros(horizon, batch_size)
    lambda_returns = torch.ones(horizon, batch_size)
    value_logits = torch.zeros(horizon, batch_size, n_bins)
    continues = torch.zeros(horizon, batch_size)
    action_logits = torch.zeros(horizon, batch_size, n_actions)
    sampled_actions = torch.zeros(horizon, batch_size, dtype=torch.long)

    normalized_actor_loss, _critic_loss, _entropy = compute_actor_critic_losses(
        value_logits,
        values,
        lambda_returns,
        continues,
        action_logits,
        sampled_actions,
        bins,
        S=1.0,
        gamma=0.997,
        actor_entropy_coef=0.0,
        normalize_advantages=True,
    )
    raw_actor_loss, _critic_loss, _entropy = compute_actor_critic_losses(
        value_logits,
        values,
        lambda_returns,
        continues,
        action_logits,
        sampled_actions,
        bins,
        S=1.0,
        gamma=0.997,
        actor_entropy_coef=0.0,
        normalize_advantages=False,
    )

    assert normalized_actor_loss.item() == 0.0
    assert raw_actor_loss.item() > 0.0


def test_actor_advantage_uses_slow_critic_baseline():
    bins = torch.linspace(-2.0, 2.0, 5)
    value_logits = torch.zeros(1, 1, 5)
    online_values = torch.full((1, 1), 100.0)
    slow_values = torch.zeros(1, 1)
    lambda_returns = torch.ones(1, 1)
    continues = torch.zeros(1, 1)
    action_logits = torch.zeros(1, 1, 2)
    sampled_actions = torch.zeros(1, 1, dtype=torch.long)

    actor_loss, _critic_loss, _entropy = compute_actor_critic_losses(
        value_logits,
        online_values,
        lambda_returns,
        continues,
        action_logits,
        sampled_actions,
        bins,
        S=1.0,
        gamma=1.0,
        actor_entropy_coef=0.0,
        normalize_advantages=False,
        actor_baseline_values=slow_values,
    )

    assert actor_loss.item() > 0.0


def test_observation_decoder_skips_state_head_when_n_observations_zero():
    mlp_cfg = SimpleNamespace(hidden_dim_ratio=8)
    cnn_cfg = SimpleNamespace(
        input_channels=3,
        kernel_size=2,
        stride=2,
        num_layers=4,
        final_feature_size=4,
    )
    env_cfg = SimpleNamespace(n_observations=0)

    decoder = ObservationDecoder(
        d_in=128,
        mlp_config=mlp_cfg,
        cnn_config=cnn_cfg,
        env_config=env_cfg,
        d_hidden=64,
    )
    out = decoder(torch.randn(2, 128))

    assert "pixels" in out
    assert "state" not in out
