import pytest
import torch

from src.trainer import twohot_encode, symexp, symlog


def test_twohot_encode():
    B = torch.arange(start=-20, end=21)
    B = symexp(B)

    bsz = 15
    x = torch.rand(bsz)

    weights = twohot_encode(x, B)

    # Check that the output shape is correct
    assert weights.shape == (bsz, len(B))
    # Check that the weights for each sample sum to 1
    assert torch.allclose(weights.sum(dim=1), torch.ones(bsz))