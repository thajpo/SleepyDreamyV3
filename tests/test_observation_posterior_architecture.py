import torch

from dreamer.config import Config
from dreamer.models import initialize_world_model


def _models(vector_encoder_mode: str, posterior_head_layers: int):
    cfg = Config(
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=2,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        vector_encoder_mode=vector_encoder_mode,
        posterior_head_layers=posterior_head_layers,
    )
    return initialize_world_model("cpu", cfg, batch_size=3)


def test_legacy_observation_posterior_preserves_checkpoint_layout():
    encoder, world_model = _models("legacy", 0)
    encoder_state = encoder.state_dict()
    world_model_state = world_model.state_dict()

    assert "MLP.mlp.0.weight" in encoder_state
    assert "MLP.mlp.2.weight" in encoder_state
    assert "MLP.mlp.4.weight" in encoder_state
    assert "MLP.mlp.1.weight" not in encoder_state
    assert isinstance(world_model.posterior_head, torch.nn.Linear)
    assert "posterior_head.weight" in world_model_state
    assert "posterior_head.0.weight" not in world_model_state


def test_reference_observation_posterior_matches_pinned_topology():
    encoder, world_model = _models("reference", 1)

    assert [type(layer) for layer in encoder.MLP.mlp] == [
        torch.nn.Linear,
        torch.nn.RMSNorm,
        torch.nn.SiLU,
    ] * 3
    assert all(
        layer.eps == 1e-4
        for layer in encoder.MLP.mlp
        if isinstance(layer, torch.nn.RMSNorm)
    )
    assert isinstance(world_model.posterior_head, torch.nn.Sequential)
    assert [type(layer) for layer in world_model.posterior_head] == [
        torch.nn.Linear,
        torch.nn.RMSNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
    ]
    assert world_model.posterior_head[1].eps == 1e-4


def test_reference_observation_posterior_keeps_end_to_end_gradients():
    encoder, world_model = _models("reference", 1)
    observations = torch.randn(3, 4, requires_grad=True)
    deter = torch.randn(3, 32, requires_grad=True)

    logits = world_model.compute_posterior(deter, encoder(observations))
    logits.square().mean().backward()

    assert logits.shape == (3, 4, 1)
    assert observations.grad is not None and observations.grad.norm().item() > 0
    assert deter.grad is not None and deter.grad.norm().item() > 0
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(
        parameter.grad is not None
        for parameter in world_model.posterior_head.parameters()
    )
