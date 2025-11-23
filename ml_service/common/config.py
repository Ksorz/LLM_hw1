"""Configuration helpers shared by both training and inference layers."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

from lib.config import (
    BASE_TRAINING_CONFIG,
    BEST_HP,
    SWEEP_CONFIG,
    build_training_config as _build_training_config,
)

__all__ = [
    "BASE_TRAINING_CONFIG",
    "BEST_HP",
    "SWEEP_CONFIG",
    "build_training_config",
]


def build_training_config(
    *,
    use_best_hp: bool = True,
    overrides: Optional[Mapping[str, float]] = None,
) -> MutableMapping[str, float]:
    """Return a mutable config dictionary ready to be consumed by training code."""

    return _build_training_config(use_best_hp=use_best_hp, overrides=overrides)
