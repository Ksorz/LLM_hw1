#!/usr/bin/env python3
"""Entrypoint для запуска ML-сервиса."""

import argparse
import logging
import os
from datetime import datetime
from typing import Dict

import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

from lib.inference import load_model_for_inference
from ml_service import create_app
from ml_service.backend.dependencies import AppDependencies
from ml_service.backend.database import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Запуск ML-сервиса")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Путь к checkpoint обученной модели (если не указан, используется необученная модель)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Путь к tokenizer (по умолчанию используется checkpoint path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство для инференса (cuda/cpu, по умолчанию auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Максимальное количество генерируемых токенов",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host для сервера",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port для сервера",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Включить auto-reload (для разработки)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    LOGGER.info("=" * 80)
    LOGGER.info("ЗАПУСК ML-СЕРВИСА")
    LOGGER.info("=" * 80)
    LOGGER.info("Checkpoint: %s", args.checkpoint or "не указан (используется необученная модель)")
    LOGGER.info("Device: %s", args.device or "auto")
    LOGGER.info("Max new tokens: %d", args.max_new_tokens)
    LOGGER.info("=" * 80)

    # Инициализация БД
    LOGGER.info("Инициализация базы данных...")
    try:
        Base.metadata.create_all(bind=engine)
        LOGGER.info("База данных инициализирована успешно.")
    except Exception as e:
        LOGGER.warning(f"Ошибка при инициализации БД (возможно, сервис БД еще не готов): {e}")
    
    # Загружаем модель
    inference_service = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Создаём функции для API
    def predict(text: str) -> str:
        return inference_service.predict(text)
    
    def predict_batch(texts):
        return inference_service.predict_batch(texts)
    
    def metadata() -> Dict[str, str]:
        return {
            "commit": os.getenv("GIT_COMMIT", "unknown"),
            "date": datetime.now().isoformat(),
            "experiment": os.path.basename(args.checkpoint) if args.checkpoint else "baseline_untrained",
            "checkpoint": args.checkpoint or "none",
            "device": str(inference_service.device),
        }
    
    # Создаём приложение
    deps = AppDependencies(
        predict=predict,
        predict_batch=predict_batch,
        metadata=metadata,
    )
    app = create_app(deps)

    # Подключаем Prometheus metrics
    Instrumentator().instrument(app).expose(app)
    
    LOGGER.info("Сервис готов к работе!")
    LOGGER.info("Swagger UI: http://%s:%d/docs", args.host, args.port)
    LOGGER.info("Metrics: http://%s:%d/metrics", args.host, args.port)
    
    # Запускаем сервер
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
