"""High level utilities for assembling the full training workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from lib import LR_MAX, LR_MIN
from lib.callbacks import InspectCallback, TimeoutCallback
from lib.modeling import create_model
from lib.optim import build_optimizer
from lib.schedulers import build_custom_scheduler_v2
from lib.tokenizer import build_tokenizer
from lib.training import build_training_arguments, create_trainer
from lib.utils.wandb_utils import build_run_name, initialize_wandb, seed_everything

from ..common.constants import MAX_TRAINING_TIME_SECONDS, OUTPUT_DIR, VALIDATION_SIZE
from ..common.config import build_training_config
from ..data.processing import (
    load_prepared_dataset,
    prepare_dataset,
    split_prepared_dataset,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "TrainingArtifacts",
    "build_training_artifacts",
]


@dataclass
class TrainingArtifacts:
    """Container that groups together all training dependencies."""

    config: Dict[str, Any]
    tokenizer: Any
    train_dataset: Any
    eval_dataset: Any
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    training_args: Any
    trainer: Any
    run_name: str
    deepspeed_config: Optional[Mapping[str, Any]]
    fsdp: Optional[str]
    fsdp_config: Optional[Mapping[str, Any]]


def _maybe_prepare_dataset(prepare_if_needed: bool, tokenizer, output_dir: str) -> None:
    if not prepare_if_needed:
        return
    LOGGER.info("Запуск подготовки датасета")
    prepare_dataset(tokenizer=tokenizer, output_dir=output_dir)


def _compose_deepspeed_config(
    training_config: Mapping[str, Any],
    *,
    stage: Optional[int],
    config_kwargs: Optional[Mapping[str, Any]],
):
    from lib.deepspeed import build_deepspeed_config

    if stage is None:
        return None

    kwargs = dict(config_kwargs or {})
    kwargs.setdefault(
        "train_micro_batch_size_per_gpu", training_config["per_device_train_batch_size"]
    )
    kwargs.setdefault("gradient_accumulation_steps", training_config["gradient_accumulation_steps"])
    return build_deepspeed_config(stage=stage, **kwargs)


def _compose_fsdp_config(fsdp_kwargs: Optional[Mapping[str, Any]]):
    from lib.fsdp import build_fsdp_config

    if not fsdp_kwargs:
        return None, None
    fsdp_dict = build_fsdp_config(**fsdp_kwargs)
    return fsdp_dict["fsdp"], fsdp_dict["fsdp_config"]


def build_training_artifacts(
    *,
    config_overrides: Optional[Mapping[str, Any]] = None,
    use_best_hp: bool = True,
    tokenizer=None,
    prepare_data_if_missing: bool = False,
    data_dir: str = OUTPUT_DIR,
    validation_size: int = VALIDATION_SIZE,
    scheduler_fn=build_custom_scheduler_v2,
    scheduler_kwargs: Optional[Mapping[str, Any]] = None,
    deepspeed_stage: Optional[int] = None,
    deepspeed_config_kwargs: Optional[Mapping[str, Any]] = None,
    fsdp_kwargs: Optional[Mapping[str, Any]] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    initialize_wandb_run: bool = True,
    wandb_project: Optional[str] = None,
    run_name_suffix: Optional[str] = None,
    timeout_seconds: float = MAX_TRAINING_TIME_SECONDS,
) -> TrainingArtifacts:
    """Collect all training dependencies in a single call."""

    config = build_training_config(use_best_hp=use_best_hp, overrides=config_overrides)
    seed_everything(int(config.get("seed", 42)))

    tokenizer = tokenizer or build_tokenizer()
    _maybe_prepare_dataset(prepare_data_if_missing, tokenizer, data_dir)

    dataset = load_prepared_dataset(data_dir)
    train_ds, eval_ds = split_prepared_dataset(dataset, validation_size=validation_size)

    model = create_model(tokenizer, model_config=model_config, bf16=config["bf16"])
    if fsdp_kwargs and fsdp_kwargs.get("activation_checkpointing", False):
        config["gradient_checkpointing"] = False
    else:
        config["gradient_checkpointing"] = True
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    optimizer = build_optimizer(
        model,
        lr=float(config.get("learning_rate", LR_MIN)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    sched_kwargs = {"lr_min": LR_MIN, "lr_max": LR_MAX}
    if scheduler_kwargs:
        sched_kwargs.update(scheduler_kwargs)
    scheduler = scheduler_fn(optimizer=optimizer, **sched_kwargs)

    deepspeed_config = _compose_deepspeed_config(
        config,
        stage=deepspeed_stage,
        config_kwargs=deepspeed_config_kwargs,
    )
    fsdp_string, fsdp_config = _compose_fsdp_config(fsdp_kwargs)

    training_args = build_training_arguments(
        config,
        deepspeed_config=deepspeed_config,
        fsdp=fsdp_string,
        fsdp_config=fsdp_config,
    )
    if deepspeed_config is not None:
        LOGGER.info("Deepspeed config:\n%s", deepspeed_config)
    if fsdp_string is not None:
        LOGGER.info("FSDP:\n%s", fsdp_string)
    if fsdp_config is not None:
        LOGGER.info("FSDP config:\n%s", fsdp_config)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_args=training_args,
        optimizer=optimizer,
        scheduler=scheduler,
        timeout_seconds=timeout_seconds,
        callbacks=[TimeoutCallback(timeout_seconds=timeout_seconds), InspectCallback()],
    )

    run_name = build_run_name(config, suffix=run_name_suffix)
    if initialize_wandb_run:
        initialize_wandb(project=wandb_project, run_name=run_name, config=config)

    return TrainingArtifacts(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_args=training_args,
        trainer=trainer,
        run_name=run_name,
        deepspeed_config=deepspeed_config,
        fsdp=fsdp_string,
        fsdp_config=fsdp_config,
    )
