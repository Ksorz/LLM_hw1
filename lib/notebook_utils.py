"""Утилиты для работы с notebook_launcher."""

import os
import torch.multiprocessing as mp


def setup_multiprocessing_for_notebook():
    """Настроить multiprocessing для работы с CUDA в notebook_launcher.
    
    Вызывайте эту функцию в самом начале ноутбука, до любых импортов torch/CUDA.
    """
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
            print("✓ Multiprocessing start method установлен в 'spawn'")
        except RuntimeError as e:
            print(f"⚠ Не удалось установить start method: {e}")
            print("  Перезапустите ядро ноутбука и попробуйте снова")


def get_distributed_env_info():
    """Получить информацию о распределённом окружении."""
    import torch.distributed as dist
    
    info = {
        "rank": int(os.getenv("RANK", "-1")),
        "local_rank": int(os.getenv("LOCAL_RANK", "-1")),
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
        "is_distributed": dist.is_available() and dist.is_initialized(),
    }
    
    if info["is_distributed"]:
        info["dist_rank"] = dist.get_rank()
        info["dist_world_size"] = dist.get_world_size()
    
    return info


__all__ = [
    "setup_multiprocessing_for_notebook",
    "get_distributed_env_info",
]

