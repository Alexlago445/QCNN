from __future__ import annotations

import math

import torch


class SPSA(torch.optim.Optimizer):
    """Two-sided SPSA with one simultaneous Rademacher direction per step.

    The gain sequences are a_k = lr / k**alpha and c_k = c / k**gamma,
    starting at k=1 (stability constant A=0). The closure must return the
    scalar loss on the same mini-batch for both perturbed evaluations.
    """

    def __init__(self, params, *, lr=0.1, c=0.1, alpha=0.602, gamma=0.101):
        for name, value in (("lr", lr), ("c", c), ("alpha", alpha), ("gamma", gamma)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SPSA {name} must be finite and positive.")
        params = [parameter for parameter in params if parameter.requires_grad]
        super().__init__(params, dict(lr=lr, c=c, alpha=alpha, gamma=gamma))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SPSA requires a loss closure.")

        group = self.param_groups[0]
        params = group["params"]
        state = self.state[params[0]]
        iteration = state.get("step", 0) + 1
        learning_rate = group["lr"] / iteration ** group["alpha"]
        perturbation = group["c"] / iteration ** group["gamma"]
        originals = [parameter.detach().clone() for parameter in params]
        directions = [torch.empty_like(parameter).bernoulli_(0.5).mul_(2).sub_(1)
                      for parameter in params]

        # Always restore the unperturbed parameters, including if evaluation fails.
        try:
            for parameter, original, direction in zip(params, originals, directions):
                parameter.copy_(original + perturbation * direction)
            loss_plus = closure().detach().clone()
            for parameter, original, direction in zip(params, originals, directions):
                parameter.copy_(original - perturbation * direction)
            loss_minus = closure().detach().clone()
        finally:
            for parameter, original in zip(params, originals):
                parameter.copy_(original)

        difference = ((loss_plus - loss_minus) / (2 * perturbation)).item()
        if not math.isfinite(difference):
            raise RuntimeError("SPSA encountered a non-finite loss difference.")
        for parameter, direction in zip(params, directions):
            parameter.add_(direction, alpha=-learning_rate * difference)
        state["step"] = iteration
        return (loss_plus + loss_minus) / 2
