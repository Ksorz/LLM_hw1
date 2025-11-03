"""Пользовательские callback'и для Trainer."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch.distributed as dist
from transformers import TrainerCallback
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType


LOGGER = logging.getLogger(__name__)


class TimeoutCallback(TrainerCallback):
    """Останавливает обучение по таймауту, поддерживая распределённый режим."""

    def __init__(self, timeout_seconds: float, check_every_n_steps: int = 1):
        self.timeout_seconds = float(timeout_seconds)
        self.check_every_n_steps = int(max(check_every_n_steps, 1))
        self.start_time: Optional[float] = None
        self.step = 0
        self.is_distributed = False
        self.rank = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.check_every_n_steps != 0:
            return control

        local_stop = False
        if self.rank == 0:
            local_stop = (time.monotonic() - (self.start_time or 0.0)) > self.timeout_seconds

        should_stop = local_stop
        if self.is_distributed:
            flag = [should_stop]
            dist.broadcast_object_list(flag, src=0)
            should_stop = bool(flag[0])

        if should_stop:
            control.should_training_stop = True
            if self.rank == 0:
                elapsed = time.monotonic() - (self.start_time or 0.0)
                LOGGER.info("Принудительная остановка обучения по таймауту: %.2f c", elapsed)

        return control


class InspectCallback(TrainerCallback):
    """Колбэк для инспекции обёртки модели (DeepSpeed/FSDP/DDP)."""
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Логировать информацию о типе обёртки модели при начале обучения."""
        
        if model is None:
            return control
        
        # Проверяем distributed type
        try:
            accelerator_state = AcceleratorState()
            distributed_type = accelerator_state.distributed_type
            LOGGER.info(f"Distributed type: {distributed_type}")
        except Exception as e:
            LOGGER.warning(f"Не удалось получить AcceleratorState: {e}")
        
        # Проверяем тип обёртки модели
        model_type = type(model).__name__
        model_module = getattr(model, '__module__', 'unknown')
        
        LOGGER.info(f"Model wrapper type: {model_type} (module: {model_module})")
        
        # Дополнительная информация для DeepSpeed
        if model_type == "DeepSpeedEngine" or "deepspeed" in model_module.lower():
            try:
                if hasattr(model, 'zero_optimization_stage'):
                    stage = model.zero_optimization_stage()
                    LOGGER.info(f"DeepSpeed ZeRO stage: {stage}")
            except Exception:
                pass
            
            try:
                if hasattr(model, 'config'):
                    zero_config = model.config.get('zero_optimization', {})
                    stage = zero_config.get('stage', 'unknown')
                    LOGGER.info(f"DeepSpeed ZeRO stage (from config): {stage}")
            except Exception:
                pass
        
        # Дополнительная информация для FSDP
        if "FSDP" in model_type or "fsdp" in model_module.lower():
            LOGGER.info("Модель обёрнута в FSDP")
            try:
                if hasattr(model, 'sharding_strategy'):
                    LOGGER.info(f"FSDP sharding strategy: {model.sharding_strategy}")
            except Exception:
                pass
        
        return control


__all__ = ["TimeoutCallback", "InspectCallback"]


