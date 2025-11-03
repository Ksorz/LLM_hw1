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
    attn_implementation: str = "flash_attention_2",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Инстанциировать Qwen3ForCausalLM из конфигурации."""

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
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )

    if LOGGER.isEnabledFor(logging.INFO):
        with torch.no_grad():
            total_params = sum(p.numel() for p in model.parameters())
        LOGGER.info("Инициализирована модель: %s параметров", f"{total_params:,}")

    return model


__all__ = ["create_model", "DEFAULT_MODEL_CONFIG"]


