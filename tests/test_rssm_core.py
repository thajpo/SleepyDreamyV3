import torch
import torch.nn.functional as F

from dreamer.config import Config
from dreamer.models import initialize_world_model


def _world_model(rssm_core: str):
    cfg = Config(
        d_hidden=16,
        num_latents=4,
        rnn_n_blocks=2,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        rssm_core=rssm_core,
    )
    return initialize_world_model("cpu", cfg, batch_size=3)[1]


def test_legacy_rssm_core_preserves_historical_parameter_layout():
    state = _world_model("legacy").state_dict()

    assert "_W_ir" in state
    assert "z_embedding.weight" in state
    assert "dynin_deter.0.weight" not in state


def test_reference_rssm_core_uses_normalized_grouped_layout():
    state = _world_model("reference").state_dict()

    assert "dynin_deter.0.weight" in state
    assert "z_embedding.0.weight" in state
    assert "dynhid.weight" in state
    assert "dynhid_norm.weight" in state
    assert "dyngru.weight" in state
    assert "_W_ir" not in state


def test_reference_rssm_core_matches_explicit_per_block_equation():
    torch.manual_seed(7)
    world_model = _world_model("reference")
    batch_size = 3
    h_prev = torch.randn(batch_size, 2 * 16, requires_grad=True)
    z_flat = torch.randn(batch_size, 4 * 1, requires_grad=True)
    action = torch.randn(batch_size, 2, requires_grad=True)
    z_embed = world_model.z_embedding(z_flat)

    actual_h, _ = world_model.step_dynamics(z_embed, action, h_prev)

    h_blocks = h_prev.reshape(batch_size, 2, 16)
    shared = torch.cat(
        (world_model.dynin_deter(h_prev), z_embed, world_model.dynin_action(action)),
        dim=-1,
    )
    hidden_blocks = []
    for block in range(2):
        block_input = torch.cat((h_blocks[:, block], shared), dim=-1)
        hidden_blocks.append(
            F.linear(
                block_input,
                world_model.dynhid.weight[block],
                world_model.dynhid.bias[block],
            )
        )
    hidden = torch.cat(hidden_blocks, dim=-1)
    hidden = F.silu(world_model.dynhid_norm(hidden)).reshape(batch_size, 2, 16)
    gates = []
    for block in range(2):
        gates.append(
            F.linear(
                hidden[:, block],
                world_model.dyngru.weight[block],
                world_model.dyngru.bias[block],
            )
        )
    gate_logits = torch.stack(gates, dim=1)
    reset_logits, candidate_logits, update_logits = gate_logits.chunk(3, dim=-1)
    reset = torch.sigmoid(reset_logits)
    candidate = torch.tanh(reset * candidate_logits)
    update = torch.sigmoid(update_logits - 1.0)
    expected_h = (update * candidate + (1.0 - update) * h_blocks).reshape(
        batch_size, -1
    )

    torch.testing.assert_close(actual_h, expected_h)
    actual_h.square().mean().backward()
    assert h_prev.grad is not None and h_prev.grad.norm().item() > 0.0
    assert z_flat.grad is not None and z_flat.grad.norm().item() > 0.0
    assert action.grad is not None and action.grad.norm().item() > 0.0
