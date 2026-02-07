from types import SimpleNamespace

import torch

from dreamer.models.decoder import ObservationDecoder
from dreamer.models.losses import compute_wm_loss


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
    posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
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
        posterior_dist=posterior_dist,
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
