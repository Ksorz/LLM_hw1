#!/usr/bin/env python3
"""Скрипт для распределённого обучения с DeepSpeed/FSDP."""

import argparse
import logging
import os
import sys
from pathlib import Path
from lib.constants import OUTPUT_DIR

from dotenv import load_dotenv

# Загружаем .env
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Распределённое обучение модели")
    
    # Режим обучения
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "deepspeed", "fsdp"],
        help="Режим обучения: baseline (single GPU), deepspeed или fsdp"
    )
    
    # DeepSpeed параметры
    parser.add_argument(
        "--deepspeed-stage",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="DeepSpeed ZeRO stage (1, 2 или 3)"
    )
    parser.add_argument(
        "--offload-optimizer",
        action="store_true",
        help="Offload optimizer на CPU (DeepSpeed)"
    )
    parser.add_argument(
        "--offload-params",
        action="store_true",
        help="Offload parameters на CPU (DeepSpeed stage 3)"
    )
    
    # Параметры обучения
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size на одно устройство"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Максимальное количество шагов (None = полная эпоха)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30 * 60,
        help="Таймаут обучения в секундах (по умолчанию 30 минут)"
    )
    
    # W&B параметры
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Имя рана в W&B"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Отключить W&B логирование"
    )
    
    # Пути
    parser.add_argument(
        "--data-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Путь к токенизированному датасету"
    )
    
    # Распределение
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Порт для distributed rendezvous (по умолчанию 29500)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info(f"Запуск обучения в режиме: {args.mode}")
    logger.info("=" * 60)
    
    # Проверяем окружение
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'не установлена')}")
        logger.info(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT', 'не установлена')}")
    
    # Импорт после настройки окружения
    from solution import (
        build_baseline_setup,
        build_deepspeed_setup,
        build_fsdp_setup,
        run_training_session,
    )
    
    # Общие параметры конфигурации
    config_overrides = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "bf16": True,
        "report_to": "none" if args.no_wandb else "wandb",
    }
    
    if args.max_steps is not None:
        config_overrides["max_steps"] = args.max_steps
    
    # Создаём setup в зависимости от режима
    if args.mode == "baseline":
        if rank == 0:
            logger.info("Создание baseline setup (single GPU)")
        print("--------------------------------")
        print(config_overrides)
        print("--------------------------------")
        print(args)
        print("--------------------------------")
        config_overrides["torch_compile"] = False
        setup = build_baseline_setup(
            # config_overrides=config_overrides,
            run_name_suffix=args.run_name or "baseline",
            # initialize_wandb_run=not args.no_wandb and rank == 0,
            # data_dir=args.data_dir,
            # timeout_seconds=args.timeout,
        )
    
    elif args.mode == "deepspeed":
        if rank == 0:
            logger.info(f"Создание DeepSpeed setup (stage {args.deepspeed_stage})")
        
        deepspeed_kwargs = {
            "offload_optimizer": args.offload_optimizer,
            "offload_parameters": args.offload_params,
        }
        
        setup = build_deepspeed_setup(
            stage=args.deepspeed_stage,
            config_overrides=config_overrides,
            deepspeed_config_kwargs=deepspeed_kwargs,
            run_name_suffix=args.run_name or f"ds_stage{args.deepspeed_stage}",
            initialize_wandb_run=not args.no_wandb and rank == 0,
            data_dir=args.data_dir,
            timeout_seconds=args.timeout,
        )
    
    elif args.mode == "fsdp":
        if rank == 0:
            logger.info("Создание FSDP setup")
        
        fsdp_kwargs = {
            "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
        }
        
        setup = build_fsdp_setup(
            fsdp_kwargs=fsdp_kwargs,
            config_overrides=config_overrides,
            run_name_suffix=args.run_name or "fsdp",
            initialize_wandb_run=not args.no_wandb and rank == 0,
            data_dir=args.data_dir,
            timeout_seconds=args.timeout,
        )
    
    if rank == 0:
        logger.info("Setup создан успешно")
        logger.info(f"Конфигурация: {setup['config']}")
    
    # Запуск обучения
    if rank == 0:
        logger.info("Старт обучения...")
    
    metrics = run_training_session(setup, final_evaluation=True)
    
    if rank == 0:
        logger.info("Обучение завершено!")
        logger.info(f"Метрики: {metrics}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

