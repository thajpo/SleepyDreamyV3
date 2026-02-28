"""Custom optimizers for DreamerV3 training."""

import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    """
    LaProp optimizer as described in DreamerV3.

    The paper describes this as "RMSProp followed by momentum":
      1. Compute second moment via EMA: v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
      2. Normalize gradient: g_norm = g / sqrt(v_t + ε)
      3. Apply momentum to normalized gradient: m_t = β₁ * m_{t-1} + (1-β₁) * g_norm
      4. Update: θ -= lr * m_t

    Key insight: the momentum is applied to the NORMALIZED gradient, not the
    raw gradient. This means the first moment tracks a smoothed version of
    the gradient direction (unit-scale), not the raw gradient magnitude.

    Used in DreamerV3 with β₁=0.9, β₂=0.999, ε=1e-20, AGC=0.3.
    """

    def __init__(
        self,
        params,
        lr: float = 4e-5,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-20,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Perform a single optimization step."""
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Standard EMA second moment (NOT max-based)
                # v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute normalized gradient (RMSProp-style)
                # No bias correction on second moment — matches source's scale_by_rms
                denom = exp_avg_sq.sqrt().add_(eps)
                normalized_grad = grad / denom

                # Momentum on the normalized gradient (not the raw gradient)
                # m_t = β₁ * m_{t-1} + (1-β₁) * g_norm
                exp_avg.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)

                # Update parameters
                p.add_(exp_avg, alpha=-lr)


def adaptive_gradient_clipping(parameters, clip_factor: float = 0.3, eps: float = 1e-3):
    """
    Adaptive Gradient Clipping (AGC) from NFNet paper.

    Clips gradients based on the ratio of gradient norm to parameter norm,
    which is more stable than fixed threshold clipping.

    If ||g|| > clip_factor * ||w||, then g = g * clip_factor * ||w|| / ||g||

    Paper: https://arxiv.org/abs/2102.06171
    DreamerV3 uses clip_factor=0.3

    Args:
        parameters: Iterable of parameters to clip
        clip_factor: Maximum allowed ratio of grad norm to param norm
        eps: Small value to avoid division by zero for small parameters
    """
    for p in parameters:
        if p.grad is None:
            continue

        # Compute norms
        param_norm = p.data.norm(2)
        grad_norm = p.grad.data.norm(2)

        # Compute max allowed gradient norm
        # Use eps as minimum param norm to handle near-zero parameters
        max_grad_norm = clip_factor * torch.maximum(param_norm, torch.tensor(eps, device=p.device))

        # Clip if gradient norm exceeds threshold
        if grad_norm > max_grad_norm:
            p.grad.data.mul_(max_grad_norm / (grad_norm + 1e-8))
