"""Модуль для извлечения данных из W&B по каждому рану из лог-файлов."""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import wandb
from wandb.apis.public import Api


def extract_run_names_from_logs(logs_dir: str = "/app/data/logs") -> Dict[str, str]:
    """
    Извлечь run_name и project из лог-файлов.
    
    Returns:
        Dict[str, str]: {log_filename: run_name}
    """
    logs_dir_path = Path(logs_dir)
    if not logs_dir_path.exists():
        raise ValueError(f"Директория {logs_dir} не существует")
    
    results = {}
    
    for log_file in logs_dir_path.glob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Извлекаем run_name из строки "wandb: Syncing run ..."
            run_name_match = re.search(r"wandb: Syncing run\s+(\S+)", content)
            if run_name_match:
                run_name = run_name_match.group(1)
                results[log_file.name] = run_name
            else:
                print(f"⚠️  Не найден run_name в {log_file.name}")
        except Exception as e:
            print(f"❌ Ошибка при чтении {log_file.name}: {e}")
    
    return results


def get_wandb_project_from_logs(logs_dir: str = "/app/data/logs") -> Optional[str]:
    """Извлечь WANDB_PROJECT из лог-файлов."""
    logs_dir_path = Path(logs_dir)
    
    for log_file in logs_dir_path.glob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "WANDB_PROJECT:" in line:
                        match = re.search(r"WANDB_PROJECT:\s*(\S+)", line)
                        if match:
                            return match.group(1)
        except Exception:
            continue
    
    # Если не нашли в логах, пробуем из env
    return os.getenv("WANDB_PROJECT", "llm_hw2-aylesnov")


def fetch_run_data(api: Api, project: str, run_name: str) -> Optional[Dict[str, Any]]:
    """Получить все данные для одного рана из W&B."""
    try:
        # Ищем run по имени в проекте
        runs = api.runs(f"{api.entity}/{project}", filters={"display_name": run_name})
        
        if not runs:
            print(f"⚠️  Run '{run_name}' не найден в проекте {project}")
            return None
        
        run = runs[0]  # Берем первый найденный
        
        # Собираем все данные
        data = {
            # Базовая информация
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "runtime": run.summary.get("_runtime", None),
            "timestamp": run.summary.get("_timestamp", None),
        }
        
        # Конфигурация (все параметры)
        if run.config:
            for key, value in run.config.items():
                # Преобразуем сложные объекты в строки для DataFrame
                if isinstance(value, (dict, list)):
                    data[f"config_{key}"] = json.dumps(value)
                else:
                    data[f"config_{key}"] = value
        
        # Метрики из summary (последние значения)
        if run.summary:
            for key, value in run.summary.items():
                if not key.startswith("_"):  # Пропускаем служебные поля
                    if isinstance(value, (dict, list)):
                        data[f"summary_{key}"] = json.dumps(value)
                    else:
                        data[f"summary_{key}"] = value
        
        # История метрик (для ключевых метрик)
        history = run.history()
        if not history.empty:
            # Берём последние значения ключевых метрик
            for col in history.columns:
                if col not in ["_step", "_timestamp", "_runtime"]:
                    last_value = history[col].dropna().iloc[-1] if not history[col].dropna().empty else None
                    if last_value is not None:
                        data[f"final_{col}"] = last_value
            
            # Статистика по ключевым метрикам
            if "train/loss" in history.columns:
                train_loss = history["train/loss"].dropna()
                if not train_loss.empty:
                    data["train_loss_min"] = float(train_loss.min())
                    data["train_loss_max"] = float(train_loss.max())
                    data["train_loss_mean"] = float(train_loss.mean())
                    data["train_loss_final"] = float(train_loss.iloc[-1])
            
            if "eval/loss" in history.columns:
                eval_loss = history["eval/loss"].dropna()
                if not eval_loss.empty:
                    data["eval_loss_min"] = float(eval_loss.min())
                    data["eval_loss_max"] = float(eval_loss.max())
                    data["eval_loss_mean"] = float(eval_loss.mean())
                    data["eval_loss_final"] = float(eval_loss.iloc[-1])
            
            # Количество шагов
            data["total_steps"] = int(history["_step"].max()) if "_step" in history.columns else None
            
            # Время обучения
            if "_runtime" in history.columns:
                runtime = history["_runtime"].dropna()
                if not runtime.empty:
                    data["training_runtime_seconds"] = float(runtime.iloc[-1])
        
        # Системные метрики (если есть)
        if hasattr(run, "system_metrics"):
            system_metrics = run.system_metrics
            if system_metrics:
                for key, value in system_metrics.items():
                    if isinstance(value, (dict, list)):
                        data[f"system_{key}"] = json.dumps(value)
                    else:
                        data[f"system_{key}"] = value
        
        return data
    
    except Exception as e:
        print(f"❌ Ошибка при получении данных для run '{run_name}': {e}")
        return None


def extract_all_runs_data(
    logs_dir: str = "/app/data/logs",
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Извлечь все данные из W&B для ранов, указанных в лог-файлах.
    
    Args:
        logs_dir: Директория с лог-файлами
        output_csv: Путь для сохранения CSV (опционально)
    
    Returns:
        pd.DataFrame: DataFrame с данными по каждому рану
    """
    print("🔍 Извлечение run_name из лог-файлов...")
    run_names = extract_run_names_from_logs(logs_dir)
    print(f"✅ Найдено {len(run_names)} ранов в логах:")
    for log_file, run_name in run_names.items():
        print(f"   {log_file} → {run_name}")
    
    print("\n🔍 Определение W&B проекта...")
    project = get_wandb_project_from_logs(logs_dir)
    print(f"✅ Проект: {project}")
    
    print("\n🔌 Подключение к W&B API...")
    api = Api()
    entity = api.entity
    print(f"✅ Подключено к entity: {entity}")
    
    print(f"\n📥 Получение данных для {len(run_names)} ранов...")
    all_data = []
    
    for log_file, run_name in run_names.items():
        print(f"   Получение данных для '{run_name}' (из {log_file})...")
        data = fetch_run_data(api, project, run_name)
        if data:
            data["log_file"] = log_file  # Добавляем имя лог-файла
            all_data.append(data)
            print(f"   ✅ Получено")
        else:
            print(f"   ⚠️  Пропущен")
    
    if not all_data:
        print("❌ Не удалось получить данные ни для одного рана")
        return pd.DataFrame()
    
    print(f"\n📊 Создание DataFrame из {len(all_data)} ранов...")
    df = pd.DataFrame(all_data)
    
    # Сортируем колонки: сначала важные, потом остальные
    important_cols = [
        "run_name", "log_file", "run_id", "state", 
        "created_at", "updated_at", "runtime", "total_steps",
        "training_runtime_seconds",
    ]
    
    # Собираем все колонки
    other_cols = [col for col in df.columns if col not in important_cols]
    
    # Сортируем другие колонки: config_*, summary_*, final_*, train_*, eval_*, system_*
    config_cols = [col for col in other_cols if col.startswith("config_")]
    summary_cols = [col for col in other_cols if col.startswith("summary_")]
    final_cols = [col for col in other_cols if col.startswith("final_")]
    train_cols = [col for col in other_cols if col.startswith("train_")]
    eval_cols = [col for col in other_cols if col.startswith("eval_")]
    system_cols = [col for col in other_cols if col.startswith("system_")]
    other_remaining = [col for col in other_cols if col not in 
                      config_cols + summary_cols + final_cols + train_cols + eval_cols + system_cols]
    
    # Переупорядочиваем колонки
    ordered_cols = (
        important_cols + 
        sorted(config_cols) + 
        sorted(summary_cols) + 
        sorted(final_cols) + 
        sorted(train_cols) + 
        sorted(eval_cols) + 
        sorted(system_cols) + 
        sorted(other_remaining)
    )
    
    # Оставляем только существующие колонки
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols]
    
    print(f"✅ DataFrame создан: {len(df)} строк, {len(df.columns)} колонок")
    
    if output_csv:
        print(f"\n💾 Сохранение в {output_csv}...")
        df.to_csv(output_csv, index=False)
        print(f"✅ Сохранено")
    
    return df


__all__ = [
    "extract_run_names_from_logs",
    "get_wandb_project_from_logs",
    "fetch_run_data",
    "extract_all_runs_data",
]