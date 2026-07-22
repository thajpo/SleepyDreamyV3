import torch
import pytest

from dreamer.config import Config
from dreamer.models import initialize_world_model


@pytest.mark.parametrize("rssm_core", ["legacy", "reference"])
def test_observe_state_keeps_gradients_across_timesteps(rssm_core):
    cfg = Config(
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=1,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        rssm_core=rssm_core,
    )
    encoder, world_model = initialize_world_model("cpu", cfg, batch_size=1)
    world_model.init_state(batch_size=1)

    first_tokens = encoder(torch.randn(1, cfg.n_observations))
    first_action = torch.tensor([[1.0, 0.0]])
    first_output = world_model(first_tokens, first_action)
    first_h = world_model.h_prev
    first_z = first_output[4]
    first_h.retain_grad()
    first_z.retain_grad()

    second_tokens = encoder(torch.randn(1, cfg.n_observations))
    second_action = torch.tensor([[0.0, 1.0]])
    second_output = world_model(second_tokens, second_action)
    second_h_z = second_output[3]
    second_h_z.sum().backward()

    assert first_h.grad is not None
    assert first_h.grad.norm().item() > 0.0
    assert first_z.grad is not None
    assert first_z.grad.norm().item() > 0.0


def test_observe_returns_raw_posterior_logits_for_single_unimix_in_loss():
    cfg = Config(
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=1,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
    )
    encoder, world_model = initialize_world_model("cpu", cfg, batch_size=1)
    captured = {}

    def capture_posterior_head(_module, _inputs, output):
        captured["raw"] = output.view(1, world_model.n_latents, world_model.n_classes)

    handle = world_model.posterior_head.register_forward_hook(capture_posterior_head)
    try:
        tokens = encoder(torch.randn(1, cfg.n_observations))
        output = world_model(tokens, torch.tensor([[1.0, 0.0]]))
    finally:
        handle.remove()

    returned_posterior_logits = output[6]
    assert torch.equal(returned_posterior_logits, captured["raw"])
