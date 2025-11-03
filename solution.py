"""Высокоуровневые утилиты для обучения модели и подготовки данных."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from transformers import Trainer

from lib import (
    LR_MAX,
    LR_MIN,
    MAX_TRAINING_TIME_SECONDS,
    OUTPUT_DIR,
    VALIDATION_SIZE,
    build_custom_scheduler_v2,
    build_deepspeed_config,
    build_fsdp_config,
    build_run_name,
    build_tokenizer,
    build_training_config,
    build_training_arguments,
    create_model,
    create_trainer,
    initialize_wandb,
    load_tokenized_dataset,
    prepare_tokenized_dataset,
    run_training,
    split_dataset,
)
from lib.optim import build_optimizer
from lib.utils import seed_everything


LOGGER = logging.getLogger(__name__)


def prepare_dataset(
    *,
    tokenizer=None,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.ru",
    split: str = "train",
    max_length: int = 512,
    num_proc: int = 4,
    output_dir: str = OUTPUT_DIR,
    num_shards: int = 32,
):
    """Подготовить parquet-шарды датасета."""

    tokenizer = tokenizer or build_tokenizer()
    return prepare_tokenized_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_length=max_length,
        num_proc=num_proc,
        output_dir=output_dir,
        num_shards=num_shards,
    )


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
    if stage is None:
        return None

    kwargs = dict(config_kwargs or {})
    kwargs.setdefault("train_micro_batch_size_per_gpu", training_config["per_device_train_batch_size"])
    kwargs.setdefault("gradient_accumulation_steps", training_config["gradient_accumulation_steps"])
    return build_deepspeed_config(stage=stage, **kwargs)


def _compose_fsdp_config(fsdp_kwargs: Optional[Mapping[str, Any]]):
    if not fsdp_kwargs:
        return None, None
    fsdp_dict = build_fsdp_config(**fsdp_kwargs)
    return fsdp_dict["fsdp"], fsdp_dict["fsdp_config"]


def build_trainer_setup(
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
) -> Dict[str, Any]:
    """Собрать все компоненты обучения в одном месте."""

    config = build_training_config(use_best_hp=use_best_hp, overrides=config_overrides)
    seed_everything(int(config.get("seed", 42)))
    
    tokenizer = tokenizer or build_tokenizer()
    _maybe_prepare_dataset(prepare_data_if_missing, tokenizer, data_dir)

    dataset = load_tokenized_dataset(data_dir)
    train_ds, eval_ds = split_dataset(dataset, validation_size=validation_size)

    model = create_model(tokenizer, model_config=model_config)
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
        LOGGER.info(f"Deepspeed config:\n{deepspeed_config}")
    if fsdp_string is not None:
        LOGGER.info(f"FSDP:\n{fsdp_string}")
    if fsdp_config is not None:
        LOGGER.info(f"FSDP config:\n{fsdp_config}")

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_args=training_args,
        optimizer=optimizer,
        scheduler=scheduler,
        timeout_seconds=timeout_seconds,
    )

    run_name = build_run_name(config, suffix=run_name_suffix)
    if initialize_wandb_run:
        initialize_wandb(project=wandb_project, run_name=run_name, config=config)

    return {
        "config": config,
        "tokenizer": tokenizer,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "training_args": training_args,
        "trainer": trainer,
        "run_name": run_name,
        "deepspeed_config": deepspeed_config,
        "fsdp": fsdp_string,
        "fsdp_config": fsdp_config,
    }


def build_baseline_setup(**kwargs) -> Dict[str, Any]:
    """Single-GPU/обычный режим"""

    return build_trainer_setup(**kwargs)


def build_deepspeed_setup(
    stage: int,
    *,
    deepspeed_config_kwargs: Optional[Mapping[str, Any]] = None,
    run_name_suffix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Сетап DeepSpeed"""

    suffix = run_name_suffix or f"ds_stage{stage}"
    return build_trainer_setup(
        deepspeed_stage=stage,
        deepspeed_config_kwargs=deepspeed_config_kwargs,
        run_name_suffix=suffix,
        **kwargs,
    )


def build_fsdp_setup(
    *,
    fsdp_kwargs: Optional[Mapping[str, Any]] = None,
    run_name_suffix: Optional[str] = "fsdp",
    **kwargs,
) -> Dict[str, Any]:
    """Сетап FSDP"""

    return build_trainer_setup(
        fsdp_kwargs=fsdp_kwargs or {},
        run_name_suffix=run_name_suffix,
        **kwargs,
    )


def run_training_session(
    trainer_or_setup,
    *,
    final_evaluation: bool = True,
    generate_text: bool = True,
    generation_prompt: str = "В начале было слово, и слово было",
) -> Dict[str, Any]:
    """Запустить обучение для Trainer или словаря из build_trainer_setup."""

    trainer = trainer_or_setup
    if not isinstance(trainer, Trainer):
        trainer = trainer_or_setup["trainer"]

    return run_training(
        trainer,
        final_evaluation=final_evaluation,
        generate_text=generate_text,
        generation_prompt=generation_prompt,
    )


__all__ = [
    "prepare_dataset",
    "build_trainer_setup",
    "build_baseline_setup",
    "build_deepspeed_setup",
    "build_fsdp_setup",
    "run_training_session",
]
