"""Фабрики конфигураций DeepSpeed."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_deepspeed_config(
    stage: int,
    *,
    bf16: bool = True,
    gradient_clipping: float = 1.0,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
    overlap_comm: bool = False,
    reduce_bucket_size: int = int(5e8),
    allgather_bucket_size: int = int(5e8),
    allgather_partitions: bool = True,
    reduce_scatter: bool = True,
    contiguous_gradients: bool = True,
    stage3_prefetch_bucket_size: int = int(5e8),
    stage3_param_persistence_threshold: int = int(1e5),
    stage3_max_live_parameters: int = int(1e9),
    stage3_max_reuse_distance: int = int(1e9),
    stage3_gather_16bit_weights_on_model_save: bool = True,
    train_micro_batch_size_per_gpu: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Сконструировать словарь DeepSpeed ZeRO Stage 1-3."""

    if stage not in (1, 2, 3):
        raise ValueError("stage должен быть 1, 2 или 3")

    zero_config: Dict[str, Any] = {
        "stage": stage,
        "overlap_comm": overlap_comm,
        "reduce_bucket_size": reduce_bucket_size,
        "allgather_bucket_size": allgather_bucket_size,
        "allgather_partitions": allgather_partitions,
        "reduce_scatter": reduce_scatter,
        "contiguous_gradients": contiguous_gradients,
    }

    if offload_optimizer and stage >= 2:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    elif offload_optimizer:
        # Stage 1 не поддерживает offload optimizer
        zero_config["offload_optimizer"] = {"device": "none"}

    if offload_parameters and stage == 3:
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}
    elif offload_parameters:
        # Offload параметров доступен только в Stage 3
        zero_config["offload_param"] = {"device": "none"}

    if stage == 3:
        zero_config.update(
            {
                "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
                "stage3_max_live_parameters": stage3_max_live_parameters,
                "stage3_max_reuse_distance": stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": stage3_gather_16bit_weights_on_model_save,
            }
        )

    cfg: Dict[str, Any] = {
        "bf16": {"enabled": bf16},
        "gradient_clipping": gradient_clipping,
        "zero_optimization": zero_config,
    }

    if train_micro_batch_size_per_gpu is not None:
        cfg["train_micro_batch_size_per_gpu"] = int(train_micro_batch_size_per_gpu)
    if gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    print("--------------------------------")
    print("Deepspeed config:")
    print(cfg)
    print("--------------------------------")
    return cfg


__all__ = ["build_deepspeed_config"]


