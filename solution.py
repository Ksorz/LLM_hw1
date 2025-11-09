"""Compatibility layer that exposes the legacy helper functions."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ml_service import build_training_artifacts, prepare_dataset as _prepare_dataset
from ml_service.common.constants import OUTPUT_DIR, VALIDATION_SIZE, MAX_TRAINING_TIME_SECONDS
from lib.schedulers import build_custom_scheduler_v2


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
    return _prepare_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_length=max_length,
        num_proc=num_proc,
        output_dir=output_dir,
        num_shards=num_shards,
    )


def build_trainer_setup(
    *,
    config_overrides: Optional[Mapping[str, Any]] = None,
    use_best_hp: bool = True,
    tokenizer=None,
    prepare_data_if_missing: bool = False,
    data_dir: str = OUTPUT_DIR,
    validation_size: int = VALIDATION_SIZE,
    scheduler_fn=None,
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
    artifacts = build_training_artifacts(
        config_overrides=config_overrides,
        use_best_hp=use_best_hp,
        tokenizer=tokenizer,
        prepare_data_if_missing=prepare_data_if_missing,
        data_dir=data_dir or OUTPUT_DIR,
        validation_size=validation_size or VALIDATION_SIZE,
        scheduler_fn=scheduler_fn or build_custom_scheduler_v2,
        scheduler_kwargs=scheduler_kwargs,
        deepspeed_stage=deepspeed_stage,
        deepspeed_config_kwargs=deepspeed_config_kwargs,
        fsdp_kwargs=fsdp_kwargs,
        model_config=model_config,
        initialize_wandb_run=initialize_wandb_run,
        wandb_project=wandb_project,
        run_name_suffix=run_name_suffix,
        timeout_seconds=timeout_seconds or MAX_TRAINING_TIME_SECONDS,
    )
    payload = artifacts.__dict__.copy()
    return payload


__all__ = ["prepare_dataset", "build_trainer_setup"]
