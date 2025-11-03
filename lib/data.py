"""Работа с датасетами: загрузка, токенизация, сохранение шардов."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from datasets import Dataset, load_dataset

from .constants import (
    ATTENTION_MASK,
    INPUT_IDS,
    LABELS,
    MAX_LENGTH,
    NUM_SHARDS,
    OUTPUT_DIR,
    VALIDATION_SIZE,
)


def prepare_raw_dataset(
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.ru",
    split: str = "train",
    text_field: str = "text",
    min_text_length: int = 1,
    load_kwargs: Optional[dict] = None,
) -> Dataset:
    """Загрузить и отфильтровать сырой датасет."""

    load_kwargs = load_kwargs or {}
    raw = load_dataset(dataset_name, dataset_config, split=split, **load_kwargs)

    if text_field and min_text_length > 0:
        raw = raw.filter(
            lambda item: item.get(text_field) is not None
            and len(item[text_field]) >= min_text_length
        )

    return raw


def _tokenize_batch(tokenizer, batch, max_length: int = MAX_LENGTH):
    texts = batch.get("text")
    if texts is None:
        return {INPUT_IDS: [], ATTENTION_MASK: [], LABELS: []}

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )

    labels = []
    for ids, mask in zip(tokenized[INPUT_IDS], tokenized[ATTENTION_MASK]):
        labels.append([token_id if m else -100 for token_id, m in zip(ids, mask)])

    tokenized[LABELS] = labels
    return tokenized


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    max_length: int = MAX_LENGTH,
    num_proc: int = 4,
    remove_columns: Optional[Iterable[str]] = None,
    desc: str = "tokenize",
) -> Dataset:
    """Преобразовать текстовый датасет в токены."""

    remove_columns = tuple(remove_columns or dataset.column_names)

    return dataset.map(
        lambda batch: _tokenize_batch(tokenizer, batch, max_length=max_length),
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc=desc,
    )


def save_dataset_shards(
    dataset: Dataset,
    *,
    output_dir: str = OUTPUT_DIR,
    num_shards: int = NUM_SHARDS,
) -> None:
    """Сохранить датасет по шартам в Parquet."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for index in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=index, contiguous=True)
        shard_path = target_dir / f"{index:05d}.parquet"
        shard.to_parquet(str(shard_path))


def prepare_tokenized_dataset(
    *,
    tokenizer,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.ru",
    split: str = "train",
    max_length: int = MAX_LENGTH,
    num_proc: int = 4,
    output_dir: str = OUTPUT_DIR,
    num_shards: int = NUM_SHARDS,
) -> Dataset:
    """Полный цикл: загрузка, токенизация и сохранение шардов."""

    raw = prepare_raw_dataset(dataset_name, dataset_config, split)
    tokenized = tokenize_dataset(
        raw,
        tokenizer,
        max_length=max_length,
        num_proc=num_proc,
        remove_columns=raw.column_names,
    )
    save_dataset_shards(tokenized, output_dir=output_dir, num_shards=num_shards)
    return tokenized


def load_tokenized_dataset(
    data_dir: str = OUTPUT_DIR,
    split: str = "train",
    columns=(INPUT_IDS, ATTENTION_MASK, LABELS),
) -> Dataset:
    """Загрузить токенизированный датасет из Parquet-шардов."""

    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Каталог {data_dir} не найден. Запустите prepare_tokenized_dataset().")

    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"В {data_dir} нет parquet файлов")

    dataset_dict = load_dataset("parquet", data_files={split: [str(f) for f in files]})
    dataset = dataset_dict[split]
    dataset = dataset.with_format("torch", columns=list(columns))
    return dataset


def split_dataset(
    dataset: Dataset,
    *,
    validation_size: int = VALIDATION_SIZE,
) -> Tuple[Dataset, Dataset]:
    """Разбить датасет на train/validation."""

    total_size = len(dataset)
    val_indices = range(min(validation_size, total_size))
    train_indices = range(min(validation_size, total_size), total_size)

    eval_ds = dataset.select(val_indices)
    train_ds = dataset.select(train_indices)
    
    print(f"Training samples:   {len(train_ds)}")
    print(f"Validation samples: {len(eval_ds)}")

    return train_ds, eval_ds


__all__ = [
    "prepare_raw_dataset",
    "tokenize_dataset",
    "save_dataset_shards",
    "prepare_tokenized_dataset",
    "load_tokenized_dataset",
    "split_dataset",
]


