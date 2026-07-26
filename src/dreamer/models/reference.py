"""Pinned DreamerV3 state-model building blocks.

These modules implement the numerical details of
``danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe`` rather than
PyTorch's superficially similar defaults.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .math_utils import symlog


REFERENCE_NORM_EPS = 1e-4
TRUNCATED_NORMAL_CORRECTION = 1.1368


class ReferenceRMSNorm(nn.Module):
    """RMSNorm with the reference's learned scale."""

    def __init__(self, features: int, eps: float = REFERENCE_NORM_EPS):
        super().__init__()
        self.features = int(features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        values = inputs.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        outputs = values * torch.rsqrt(mean_square + self.eps)
        outputs = outputs * self.weight.float()
        return outputs.to(dtype=dtype)


@torch.no_grad()
def reference_truncated_normal_(
    tensor: torch.Tensor,
    *,
    fan_in: int | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply pinned fan-in truncated-normal initialization.

    JAX samples a unit normal truncated to ``[-2, 2]`` and multiplies it by
    ``1.1368 / sqrt(fan_in)``. PyTorch's ``trunc_normal_`` accepts the same
    pre-scaling distribution, so the operations are kept in that order.
    """

    if fan_in is None:
        if tensor.ndim < 2:
            raise ValueError("fan_in is required for tensors with fewer than 2 axes")
        fan_in = int(tensor.shape[-1])
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
    tensor.mul_(TRUNCATED_NORMAL_CORRECTION * float(scale) / math.sqrt(fan_in))
    return tensor


@torch.no_grad()
def initialize_reference_module(module: nn.Module) -> None:
    """Initialize Linear-like modules with pinned reference defaults."""

    for child in module.modules():
        if isinstance(child, nn.Linear):
            reference_truncated_normal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class ReferenceMLP(nn.Module):
    """Reference hidden stack plus output projection."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        *,
        hidden_layers: int,
        outscale: float = 1.0,
        symlog_input: bool = False,
    ):
        super().__init__()
        self.symlog_input = bool(symlog_input)
        layers: list[nn.Module] = []
        in_features = int(d_in)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(in_features, d_hidden),
                    ReferenceRMSNorm(d_hidden),
                    nn.SiLU(),
                ]
            )
            in_features = int(d_hidden)
        layers.append(nn.Linear(in_features, d_out))
        self.mlp = nn.Sequential(*layers)
        initialize_reference_module(self)
        output = self.mlp[-1]
        assert isinstance(output, nn.Linear)
        if float(outscale) == 0.0:
            nn.init.zeros_(output.weight)
        elif float(outscale) != 1.0:
            with torch.no_grad():
                output.weight.mul_(float(outscale))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.symlog_input:
            inputs = symlog(inputs)
        return self.mlp(inputs)


class ReferenceFeatureMLP(nn.Module):
    """Reference normalized hidden stack without an output projection."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        *,
        hidden_layers: int,
        symlog_input: bool = False,
    ):
        super().__init__()
        self.symlog_input = bool(symlog_input)
        layers: list[nn.Module] = []
        in_features = int(d_in)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(in_features, d_hidden),
                    ReferenceRMSNorm(d_hidden),
                    nn.SiLU(),
                ]
            )
            in_features = int(d_hidden)
        self.mlp = nn.Sequential(*layers)
        initialize_reference_module(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.symlog_input:
            inputs = symlog(inputs)
        return self.mlp(inputs)
