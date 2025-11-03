"""Конфигурации обучения и утилиты для их модификации."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

from .constants import OUTPUT_DIR


BASE_TRAINING_CONFIG: Dict[str, object] = {
    "output_dir": f"{OUTPUT_DIR}/gpt2-1b-russian",
    "optim": "adamw_torch",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "save_steps": 2500,
    "save_total_limit": 2,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 5,
    "eval_steps": 2500,
    "eval_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "per_device_eval_batch_size": 4,
    "dataloader_num_workers": 4,
    "torch_compile": True,
    "report_to": "wandb",
}


SWEEP_CONFIG: Dict[str, object] = {
    "method": "bayes",
    "metric": {"name": "final_eval_loss", "goal": "minimize"},
    "parameters": {
        "per_device_train_batch_size": {"values": [1, 2, 4, 6]},
        "gradient_accumulation_steps": {"values": [2, 4, 6]},
        "learning_rate": {"values": [5e-5, 1e-4, 2e-4, 5e-4]},
        "optim": {"values": ["adamw_torch", "adamw_torch_fused", "adafactor"]},
        "lr_scheduler_type": {"values": ["linear", "cosine", "constant"]},
        "torch_compile": {"values": [True, False]},
        "warmup_ratio": {"values": [0.07, 0.1, 0.2]},
    },
}


BEST_HP: Dict[str, object] = {
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_torch",
    "torch_compile": True,
}


DEFAULT_DROP_KEYS = ("lr_scheduler_type", "warmup_steps", "warmup_ratio")


def build_training_config(
    *,
    use_best_hp: bool = True,
    overrides: Optional[Mapping[str, object]] = None,
    drop_keys: Optional[Iterable[str]] = DEFAULT_DROP_KEYS,
) -> Dict[str, object]:
    """Создать словарь TrainingArguments с учётом переопределений.

    Parameters
    ----------
    use_best_hp:
        Добавить ли оптимальные гиперпараметры поверх базовой конфигурации.
    overrides:
        Любые пользовательские значения, приоритетнее base/best конфигов.
    drop_keys:
        Ключи, которые нужно удалить из финального словаря (например, чтобы
        задать собственный шедулер).
    """

    config: MutableMapping[str, object] = deepcopy(BASE_TRAINING_CONFIG)

    if use_best_hp:
        config.update(BEST_HP)

    if overrides:
        config.update(overrides)

    if drop_keys:
        for key in drop_keys:
            config.pop(key, None)

    return dict(config)


