import torch
from typing import Iterable


def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6
) -> None:
    # Compute total norm
    total_norm = torch.sqrt(sum(
        (p.grad.data.norm(2) ** 2 for p in params if p.grad is not None)
    ) + eps)
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
