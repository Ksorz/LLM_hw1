"""Оптимизаторы."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW


def _can_use_fused_adamw() -> bool:
    return torch.cuda.is_available()


def build_optimizer(
    model,
    *,
    lr: float,
    weight_decay: float = 0.01,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """Создать AdamW с fused-вариантом при наличии CUDA."""

    kwargs = {}
    if _can_use_fused_adamw():
        kwargs["fused"] = True

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        **kwargs,
    )

    return optimizer


__all__ = ["build_optimizer"]


