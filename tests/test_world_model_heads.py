import torch

from dreamer.config import Config
from dreamer.models import initialize_world_model


def _world_model(continue_head_layers: int):
    cfg = Config(
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=1,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        continue_head_layers=continue_head_layers,
    )
    return initialize_world_model("cpu", cfg, batch_size=1)[1]


def test_historical_continuation_head_keeps_linear_checkpoint_layout():
    world_model = _world_model(continue_head_layers=0)

    assert isinstance(world_model.continue_predictor, torch.nn.Linear)
    assert "continue_predictor.weight" in world_model.state_dict()


def test_authored_continuation_head_has_reference_hidden_block():
    world_model = _world_model(continue_head_layers=1)
    head = world_model.continue_predictor

    assert isinstance(head, torch.nn.Sequential)
    assert [type(layer) for layer in head] == [
        torch.nn.Linear,
        torch.nn.RMSNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
    ]
    assert head(torch.randn(3, head[0].in_features)).shape == (3, 1)
