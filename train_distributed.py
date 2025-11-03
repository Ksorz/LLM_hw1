#!/usr/bin/env python3
"""Скрипт для распределённого обучения с DeepSpeed/FSDP."""

import argparse
import json
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
    
    # ==================== DeepSpeed параметры ====================
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
        help="Offload optimizer на CPU (DeepSpeed Stage 2+)"
    )
    parser.add_argument(
        "--offload-params",
        action="store_true",
        help="Offload parameters на CPU (DeepSpeed Stage 3)"
    )
    parser.add_argument(
        "--overlap-comm",
        action="store_true",
        help="Включить overlap_comm для DeepSpeed (увеличивает потребление памяти)"
    )
    parser.add_argument(
        "--reduce-bucket-size",
        type=int,
        default=int(5e8),
        help="Размер bucket для reduce операций (элементы, не байты)"
    )
    parser.add_argument(
        "--allgather-bucket-size",
        type=int,
        default=int(5e8),
        help="Размер bucket для allgather операций (элементы, не байты)"
    )
    parser.add_argument(
        "--stage3-prefetch-bucket-size",
        type=int,
        default=int(5e8),
        help="Размер prefetch bucket для Stage 3"
    )
    parser.add_argument(
        "--stage3-param-persistence-threshold",
        type=int,
        default=int(1e5),
        help="Порог для параметров, которые не шардируются в Stage 3"
    )
    
    # ==================== FSDP параметры ====================
    parser.add_argument(
        "--fsdp-sharding-strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"],
        help="FSDP sharding strategy"
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        help="Offload параметров FSDP на CPU"
    )
    parser.add_argument(
        "--fsdp-activation-checkpointing",
        action="store_true",
        help="Включить activation checkpointing для FSDP"
    )
    parser.add_argument(
        "--fsdp-backward-prefetch",
        type=str,
        default="BACKWARD_POST",
        choices=["BACKWARD_PRE", "BACKWARD_POST"],
        help="FSDP backward prefetch policy"
    )
    parser.add_argument(
        "--fsdp-forward-prefetch",
        action="store_true",
        help="Включить forward prefetch для FSDP"
    )
    parser.add_argument(
        "--fsdp-state-dict-type",
        type=str,
        default="FULL_STATE_DICT",
        choices=["FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"],
        help="Тип state dict для сохранения модели"
    )
    parser.add_argument(
        "--fsdp-transformer-layers",
        type=str,
        default="Qwen3DecoderLayer",
        help="Классы слоёв трансформера для FSDP wrap (через запятую)"
    )
    
    # ==================== Параметры обучения ====================
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Использовать bfloat16 (True/False, по умолчанию True)"
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Включить torch.compile (может замедлить распределённое обучение)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (если не указан, используется из config)"
    )
    
    # ==================== W&B параметры ====================
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
    
    # ==================== Пути ====================
    parser.add_argument(
        "--data-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Путь к токенизированному датасету"
    )
    
    # # ==================== Распределение ====================
    # # Аргумент для deepspeed/torchrun (игнорируется, берётся из окружения)
    # parser.add_argument(
    #     "--local_rank",
    #     type=int,
    #     default=None,
    #     help="Local rank (используется deepspeed/torchrun, обычно берётся из окружения)"
    # )
    
    # ==================== Генерация текста ====================
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Отключить генерацию текста после обучения"
    )
    parser.add_argument(
        "--generation-prompt",
        type=str,
        default="В начале было слово, и слово было",
        help="Prompt для генерации текста"
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
        "bf16": args.bf16,
        "report_to": "none" if args.no_wandb else "wandb",
        "torch_compile": args.torch_compile,
    }
    
    if args.learning_rate is not None:
        config_overrides["learning_rate"] = args.learning_rate
    
    if args.max_steps is not None:
        config_overrides["max_steps"] = args.max_steps
    
    # Создаём setup в зависимости от режима
    if args.mode == "baseline":
        if rank == 0:
            logger.info("Создание baseline setup (single GPU)")
        
        setup = build_baseline_setup(
            config_overrides=config_overrides,
            run_name_suffix=args.run_name or "baseline",
            initialize_wandb_run=not args.no_wandb and rank == 0,
            data_dir=args.data_dir,
            timeout_seconds=args.timeout,
        )
        # setup = build_baseline_setup(run_name_suffix="baseline_fc1")

    
    elif args.mode == "deepspeed":
        if rank == 0:
            logger.info(f"Создание DeepSpeed setup (stage {args.deepspeed_stage})")
            logger.info(f"  overlap_comm: {args.overlap_comm}")
            logger.info(f"  offload_optimizer: {args.offload_optimizer}")
            logger.info(f"  offload_params: {args.offload_params}")
            logger.info(f"  reduce_bucket_size: {args.reduce_bucket_size}")
            logger.info(f"  allgather_bucket_size: {args.allgather_bucket_size}")
        
        deepspeed_kwargs = {
            "offload_optimizer": args.offload_optimizer,
            "offload_parameters": args.offload_params,
            "overlap_comm": args.overlap_comm,
            "reduce_bucket_size": args.reduce_bucket_size,
            "allgather_bucket_size": args.allgather_bucket_size,
            "stage3_prefetch_bucket_size": args.stage3_prefetch_bucket_size,
            "stage3_param_persistence_threshold": args.stage3_param_persistence_threshold,
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
            logger.info(f"  sharding_strategy: {args.fsdp_sharding_strategy}")
            logger.info(f"  cpu_offload: {args.fsdp_cpu_offload}")
            logger.info(f"  activation_checkpointing: {args.fsdp_activation_checkpointing}")
            logger.info(f"  backward_prefetch: {args.fsdp_backward_prefetch}")
            logger.info(f"  forward_prefetch: {args.fsdp_forward_prefetch}")
            logger.info(f"  state_dict_type: {args.fsdp_state_dict_type}")
        
        transformer_layers = [x.strip() for x in args.fsdp_transformer_layers.split(",")]
        
        fsdp_kwargs = {
            "sharding_strategy": args.fsdp_sharding_strategy,
            "cpu_offload": args.fsdp_cpu_offload,
            "activation_checkpointing": args.fsdp_activation_checkpointing,
            "transformer_layer_cls_to_wrap": transformer_layers,
            "backward_prefetch": args.fsdp_backward_prefetch,
            "forward_prefetch": args.fsdp_forward_prefetch,
            "state_dict_type": args.fsdp_state_dict_type,
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
    
    metrics = run_training_session(
        setup, 
        final_evaluation=True,
        generate_text=not args.no_generation,
        generation_prompt=args.generation_prompt,
    )
    
    if rank == 0:
        logger.info("Обучение завершено!")
        logger.info(f"Метрики: {metrics}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

