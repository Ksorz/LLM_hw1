"""Пользовательские шедулеры learning rate."""

from __future__ import annotations

from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _set_initial_lr(optimizer: Optimizer, lr_min: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_min


def build_custom_scheduler(
    optimizer: Optimizer,
    *,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    s_up_end: int = 50,
    s_hold_end: int = 1300,
    s_down_end: int = 1800,
    cyc_step_up: int = 400,
) -> LambdaLR:
    """Кастомный scheduler с треугольными циклами после разогрева."""

    _set_initial_lr(optimizer, lr_min)
    ratio = lr_max / lr_min

    def lr_lambda(step: int) -> float:
        if step <= s_up_end:
            return 1.0 + (ratio - 1.0) * step / float(max(s_up_end, 1))
        if step <= s_hold_end:
            return ratio
        if step <= s_down_end:
            progress = (step - s_hold_end) / float(max(s_down_end - s_hold_end, 1))
            return ratio - (ratio - 1.0) * progress

        k = step - s_down_end
        period = max(2 * cyc_step_up, 1)
        phase = (k % period) / float(cyc_step_up)
        if phase <= 1.0:
            return 1.0 + (ratio - 1.0) * phase
        return ratio - (ratio - 1.0) * (phase - 1.0)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_custom_scheduler_v2(
    optimizer: Optimizer,
    *,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    s_up_end: int = 10,
    s_hold_end: int = 450,
    s_down_end: int = 2000,
) -> LambdaLR:
    """Упрощённый scheduler: разгон, плато, линейный спад."""

    _set_initial_lr(optimizer, lr_min)
    ratio = lr_max / lr_min

    def lr_lambda(step: int) -> float:
        if step <= s_up_end:
            return 1.0 + (ratio - 1.0) * step / float(max(s_up_end, 1))
        if step <= s_hold_end:
            return ratio
        if step <= s_down_end:
            progress = (step - s_hold_end) / float(max(s_down_end - s_hold_end, 1))
            return ratio - (ratio - 1.0) * progress
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_custom_scheduler", "build_custom_scheduler_v2"]


