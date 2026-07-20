"""Read-only gradient diagnostics for research runs."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class _GradientAccumulator:
    primary_sq: torch.Tensor
    auxiliary_sq: torch.Tensor
    dot: torch.Tensor
    numel: int = 0


def _parameter_group(name: str) -> str:
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("world_model.posterior_head."):
        return "posterior"
    if name.startswith("world_model.z_embedding."):
        return "recurrent"
    if name.startswith("world_model._W_") or name.startswith("world_model._b_"):
        return "recurrent"
    return "other"


def measure_gradient_alignment(
    primary_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, float]:
    """Measure two loss gradients without populating parameter ``.grad`` fields.

    Metrics only include parameters reached by both objectives. This compares
    directions on the shared representation subspace instead of diluting the
    result with decoder/head parameters that replay value loss cannot reach.
    """
    selected = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not selected:
        return {}

    parameters = [parameter for _, parameter in selected]
    primary_gradients = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    accumulators: dict[str, _GradientAccumulator] = {}
    for (name, _), primary, auxiliary in zip(
        selected, primary_gradients, auxiliary_gradients
    ):
        if primary is None or auxiliary is None:
            continue
        group = _parameter_group(name)
        primary_float = primary.detach().float()
        auxiliary_float = auxiliary.detach().float()
        for key in ("global", group):
            values = accumulators.setdefault(
                key,
                _GradientAccumulator(
                    primary_sq=torch.zeros((), device=primary.device),
                    auxiliary_sq=torch.zeros((), device=primary.device),
                    dot=torch.zeros((), device=primary.device),
                ),
            )
            values.primary_sq = (
                values.primary_sq + primary_float.square().sum()
            )
            values.auxiliary_sq = (
                values.auxiliary_sq + auxiliary_float.square().sum()
            )
            values.dot = values.dot + (
                primary_float * auxiliary_float
            ).sum()
            values.numel += primary.numel()

    result: dict[str, float] = {}
    for group, values in accumulators.items():
        primary_norm = values.primary_sq.sqrt()
        auxiliary_norm = values.auxiliary_sq.sqrt()
        prefix = f"research/gradient_alignment/{group}"
        result[f"{prefix}/wm_norm"] = float(primary_norm.item())
        result[f"{prefix}/replay_norm"] = float(auxiliary_norm.item())
        result[f"{prefix}/shared_numel"] = float(values.numel)
        if primary_norm.item() > 0.0:
            result[f"{prefix}/replay_to_wm_norm"] = float(
                (auxiliary_norm / primary_norm).item()
            )
        denominator = primary_norm * auxiliary_norm
        if denominator.item() > 0.0:
            result[f"{prefix}/cosine"] = float(
                (values.dot / denominator).item()
            )
    return result
