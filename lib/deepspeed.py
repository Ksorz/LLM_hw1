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
    overlap_comm: bool = True,
    reduce_bucket_size: int = 5 * 1024 * 1024 * 1024,
    allgather_bucket_size: int = 5 * 1024 * 1024 * 1024,
    stage3_prefetch_bucket_size: int = 2 * 1024 * 1024 * 1024,
    stage3_param_persistence_threshold: int = 1 * 1024 * 1024,
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
        "contiguous_gradients": True,
    }

    if offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    if offload_parameters:
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}

    if stage == 3:
        zero_config.update(
            {
                "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
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

    return cfg


__all__ = ["build_deepspeed_config"]


