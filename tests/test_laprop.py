import torch

from dreamer.models.optimizers import LaProp


def _reference_laprop_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DreamerV3's bias-corrected RMS-then-momentum update."""
    second_moment = beta2 * second_moment + (1 - beta2) * gradient.square()
    corrected_second_moment = second_moment / (1 - beta2**step)
    normalized_gradient = gradient / (corrected_second_moment.sqrt() + epsilon)
    momentum = beta1 * momentum + (1 - beta1) * normalized_gradient
    corrected_momentum = momentum / (1 - beta1**step)
    parameter = parameter - learning_rate * corrected_momentum
    return parameter, momentum, second_moment


def test_laprop_matches_bias_corrected_reference() -> None:
    learning_rate = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-20
    initial = torch.tensor([2.0, -3.0])
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = LaProp(
        [parameter],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )

    expected = initial.clone()
    momentum = torch.zeros_like(initial)
    second_moment = torch.zeros_like(initial)
    gradients = (torch.tensor([4.0, -2.0]), torch.tensor([-1.0, -6.0]))

    for step, gradient in enumerate(gradients, start=1):
        parameter.grad = gradient.clone()
        optimizer.step()
        expected, momentum, second_moment = _reference_laprop_step(
            expected,
            gradient,
            momentum,
            second_moment,
            step=step,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


def test_laprop_first_step_has_configured_update_scale() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = LaProp([parameter], lr=0.1, eps=1e-20)
    parameter.grad = torch.tensor([5.0])

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([-0.1]))
