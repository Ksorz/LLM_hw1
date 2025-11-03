"""Высокоуровневые помощники для обучения."""

from __future__ import annotations

import copy
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from transformers import Trainer, TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available

from .callbacks import TimeoutCallback
from .constants import MAX_TRAINING_TIME_SECONDS, OUTPUT_DIR
from .utils import is_main_process


LOGGER = logging.getLogger(__name__)


def build_training_arguments(
    config: Mapping[str, Any],
    *,
    bf16: Optional[bool] = None,
    seed: int = 42,
    logging_dir: Optional[str] = None,
    remove_unused_columns: bool = False,
    save_safetensors: bool = True,
    dataloader_pin_memory: bool = True,
    deepspeed_config: Optional[Mapping[str, Any]] = None,
    fsdp: Optional[str] = None,
    fsdp_config: Optional[Mapping[str, Any]] = None,
) -> TrainingArguments:
    """Построить TrainingArguments из словаря конфигурации."""

    cfg = dict(config)

    # if bf16 is None:
    #     bf16 = False
    #     if is_torch_bf16_gpu_available():
    bf16 = True

    cfg.setdefault("bf16", bf16)
    cfg.setdefault("seed", seed)
    cfg.setdefault("logging_dir", logging_dir or f"{OUTPUT_DIR}/logs")
    cfg.setdefault("remove_unused_columns", remove_unused_columns)
    cfg.setdefault("save_safetensors", save_safetensors)
    cfg.setdefault("dataloader_pin_memory", dataloader_pin_memory)

    if deepspeed_config is not None:
        cfg["deepspeed"] = copy.deepcopy(deepspeed_config)

    if fsdp is not None:
        cfg["fsdp"] = fsdp

    if fsdp_config is not None:
        cfg["fsdp_config"] = copy.deepcopy(fsdp_config)

    training_args = TrainingArguments(**cfg)
    return training_args


def create_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    optimizer=None,
    scheduler=None,
    timeout_seconds: float = MAX_TRAINING_TIME_SECONDS,
    callbacks: Optional[Sequence[Any]] = None,
):
    """Создать Trainer с кастомными оптимизаторами и Callback'ами."""

    callback_list = list(callbacks or [])
    if timeout_seconds:
        callback_list.append(TimeoutCallback(timeout_seconds))

    optimizers = None
    if optimizer is not None or scheduler is not None:
        if optimizer is None or scheduler is None:
            raise ValueError("Необходимо передать и optimizer, и scheduler, либо ни один")
        optimizers = (optimizer, scheduler)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callback_list,
        optimizers=optimizers,
    )
    return trainer


def generate_sample_text(
    model,
    tokenizer,
    prompt: str = "В начале было слово, и слово было",
    *,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    top_p: float = 0.9,
    temperature: float = 0.8,
) -> str:
    """Сгенерировать пример текста из модели."""
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    return generated_text


def run_training(
    trainer: Trainer,
    *,
    final_evaluation: bool = True,
    generate_text: bool = True,
    generation_prompt: str = "В начале было слово, и слово было",
) -> dict:
    """Запустить обучение и (опционально) финальную валидацию и генерацию."""

    LOGGER.info("Старт обучения")
    train_output = trainer.train()
    metrics = {"train": train_output.metrics}

    if final_evaluation:
        LOGGER.info("Финальная валидация")
        eval_metrics = trainer.evaluate()
        metrics["eval"] = eval_metrics

    if generate_text and is_main_process():
        LOGGER.info("Генерация примера текста")
        model = trainer.model
        tokenizer = trainer.tokenizer
        generated = generate_sample_text(
            model,
            tokenizer,
            prompt=generation_prompt,
        )
        LOGGER.info("\n=== SAMPLE GENERATION ===")
        LOGGER.info(generated)
        metrics["generation"] = generated

    return metrics


__all__ = [
    "build_training_arguments",
    "create_trainer",
    "run_training",
    "generate_sample_text",
]


