"""Построение модели Qwen."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIG: Dict[str, object] = {
    "hidden_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "head_dim": 128,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "scale_attn_weights": True,
    "use_cache": True,
}


def create_model(
    tokenizer,
    *,
    model_config: Optional[Dict[str, object]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    bf16: Optional[bool] = None,
):
    """Инстанциировать Qwen3ForCausalLM из конфигурации.
    
    Parameters
    ----------
    torch_dtype
        Явно указанный dtype модели. Если None, определяется из bf16.
    bf16
        Использовать ли bfloat16. Если None и torch_dtype тоже None, используется bfloat16 по умолчанию.
    """
    
    # Определяем dtype: приоритет у torch_dtype, затем bf16
    if torch_dtype is None:
        if bf16 is None:
            torch_dtype = torch.bfloat16  # По умолчанию bfloat16
        else:
            torch_dtype = torch.bfloat16 if bf16 else torch.float32

    cfg = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        cfg.update(model_config)

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **cfg,
    )

    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2' if torch_dtype == torch.bfloat16 else "sdpa",
        torch_dtype=torch_dtype,
    )

    if LOGGER.isEnabledFor(logging.INFO):
        with torch.no_grad():
            # Общее число параметров
            total_params = sum(p.numel() for p in model.parameters())
            model_dtype = next(model.parameters()).dtype

            # Вес одного параметра в байтах
            bytes_per_param = {
                torch.float32: 4,
                torch.bfloat16: 2,
                torch.float16: 2,
                torch.int8: 1,
            }.get(model_dtype, 4)

            # Примерный теоретический размер модели
            model_size_bytes = total_params * bytes_per_param
            model_size_gb = model_size_bytes / (1024 ** 3)

            # Проверяем, где живёт модель
            try:
                first_device = next(model.parameters()).device
            except StopIteration:
                first_device = torch.device("cpu")

            # GPU/CPU статус без логирования реального использования памяти
            if torch.cuda.is_available() and first_device.type == "cuda":
                location_str = f"на GPU ({first_device})"
            else:
                location_str = "на CPU (модель не перенесена на GPU)"

        LOGGER.info(
            "Модель: %s параметров, dtype=%s, ~%.2f GB, %s",
            f"{total_params:,}",
            model_dtype,
            model_size_gb,
            location_str,
        )



    return model


__all__ = ["create_model", "DEFAULT_MODEL_CONFIG"]


