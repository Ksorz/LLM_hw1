"""Фабрики конфигураций FSDP для Trainer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def build_fsdp_config(
    *,
    sharding_strategy: str = "full_shard",
    auto_wrap: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = True,
    transformer_layer_cls_to_wrap: Iterable[str] = ("Qwen3DecoderLayer",),
    backward_prefetch: str = "BACKWARD_PRE",
    state_dict_type: str = "full",
    limit_all_gathers: bool = True,
    sync_module_states: bool = True,
) -> Dict[str, Any]:
    """Сконструировать параметры для TrainingArguments fsdp/fsdp_config."""

    tokens = []
    if sharding_strategy:
        tokens.append(sharding_strategy)
    if auto_wrap:
        tokens.append("auto_wrap")
    if cpu_offload:
        tokens.append("cpu_offload")
    if activation_checkpointing:
        tokens.append("activation_checkpointing")
    if state_dict_type:
        tokens.append(f"state_dict_type_{state_dict_type}")

    fsdp_config: Dict[str, Any] = {
        "fsdp_transformer_layer_cls_to_wrap": list(transformer_layer_cls_to_wrap),
        "backward_prefetch": backward_prefetch,
        "limit_all_gathers": limit_all_gathers,
        "sync_module_states": sync_module_states,
    }

    if state_dict_type:
        fsdp_config["state_dict_type"] = state_dict_type

    return {
        "fsdp": " ".join(tokens),
        "fsdp_config": fsdp_config,
    }


__all__ = ["build_fsdp_config"]


