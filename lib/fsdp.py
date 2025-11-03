"""Фабрики конфигураций FSDP для Trainer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def build_fsdp_config(
    *,
    sharding_strategy: str = "full_shard",
    auto_wrap: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    transformer_layer_cls_to_wrap: Iterable[str] = ("Qwen3DecoderLayer",),
    backward_prefetch: str = "BACKWARD_POST",
    forward_prefetch: bool = False,
    state_dict_type: str = "FULL_STATE_DICT",
    limit_all_gathers: bool = True,
    sync_module_states: bool = True,
    use_orig_params: bool = True,
    cpu_ram_efficient_loading: bool = True,
    # min_num_params: int = int(1e8),
) -> Dict[str, Any]:
    """Сконструировать параметры для TrainingArguments fsdp/fsdp_config."""

    tokens = []
    if sharding_strategy:
        tokens.append(sharding_strategy)
    if auto_wrap:
        tokens.append("auto_wrap")
    if cpu_offload:
        tokens.append("offload")

    fsdp_config: Dict[str, Any] = {
        "fsdp_transformer_layer_cls_to_wrap": list(transformer_layer_cls_to_wrap),
        "fsdp_backward_prefetch": backward_prefetch,
        "fsdp_forward_prefetch": forward_prefetch,
        "fsdp_state_dict_type": state_dict_type,
        "fsdp_cpu_ram_efficient_loading": cpu_ram_efficient_loading,
        "fsdp_use_orig_params": use_orig_params,
        "fsdp_sync_module_states": sync_module_states,
        "limit_all_gathers": limit_all_gathers,
        # "min_num_params": min_num_params,
    }
    
    # activation_checkpointing обрабатывается через gradient_checkpointing в TrainingArguments
    if activation_checkpointing:
        fsdp_config["fsdp_activation_checkpointing"] = True

    return {
        "fsdp": " ".join(tokens),
        "fsdp_config": fsdp_config,
    }


__all__ = ["build_fsdp_config"]


