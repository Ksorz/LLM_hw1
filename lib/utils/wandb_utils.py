"""Утилиты: распределение, wandb, логирование."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

import torch.distributed as dist
import wandb
from transformers import set_seed


LOGGER = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def seed_everything(seed: int = 0) -> None:
    set_seed(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank(default: int = 0) -> int:
    if is_distributed():
        return dist.get_rank()
    return default


def is_main_process() -> bool:
    return get_rank() == 0


def initialize_wandb(
    *,
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    mode: Optional[str] = None,
) -> Optional[Any]:
    """Инициализировать wandb только на главном процессе."""

    if not is_main_process():
        return None

    if wandb.run is not None:
        if run_name:
            wandb.run.name = run_name
        if config:
            wandb.config.update(dict(config), allow_val_change=True)
        return wandb.run

    settings = wandb.Settings(
        http_proxy=os.getenv("AVITO_HTTP_PROXY"),
        https_proxy=os.getenv("AVITO_HTTPS_PROXY"),
    )

    run = wandb.init(
        project=project or os.getenv("WANDB_PROJECT", "llm_hw1"),
        name=run_name,
        config=config,
        settings=settings,
        mode=mode,
    )
    return run


def build_run_name(config: Mapping[str, Any], *, prefix: Optional[str] = None, suffix: Optional[str] = None) -> str:
    parts = []
    if prefix:
        parts.append(prefix)

    if "per_device_train_batch_size" in config:
        parts.append(f"bs{config['per_device_train_batch_size']}")
    if "gradient_accumulation_steps" in config:
        parts.append(f"ga{config['gradient_accumulation_steps']}")

    if suffix:
        parts.append(suffix)

    return "_".join(str(p) for p in parts if p)


__all__ = [
    "setup_logging",
    "seed_everything",
    "is_distributed",
    "get_rank",
    "is_main_process",
    "initialize_wandb",
    "build_run_name",
]


