from __future__ import annotations

import torch

# cost function for the QCNN models, shared by every model. 
def probability_cross_entropy(
    probs: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy directly on qml.probs output; shared by every model."""
    lp = torch.log(torch.clamp(probs, eps, 1.0))
    nll = -lp[torch.arange(len(y), device=y.device), y]
    if label_smoothing > 0:
        return (
            (1.0 - label_smoothing) * nll
            - label_smoothing * lp.mean(dim=1)
        ).mean()
    return nll.mean()
