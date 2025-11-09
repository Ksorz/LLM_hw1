"""Utilities that orchestrate dataset preparation for the training workflow."""

from __future__ import annotations

from typing import Optional

from datasets import DatasetDict

from lib import (
    OUTPUT_DIR,
    VALIDATION_SIZE,
    load_tokenized_dataset,
    prepare_tokenized_dataset,
    split_dataset,
)
from lib.tokenizer import build_tokenizer

__all__ = [
    "prepare_dataset",
    "load_prepared_dataset",
    "split_prepared_dataset",
]


def prepare_dataset(
    *,
    tokenizer=None,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.ru",
    split: str = "train",
    max_length: int = 512,
    num_proc: int = 4,
    output_dir: str = OUTPUT_DIR,
    num_shards: int = 32,
):
    """Prepare parquet shards for the configured dataset."""

    tokenizer = tokenizer or build_tokenizer()
    return prepare_tokenized_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_length=max_length,
        num_proc=num_proc,
        output_dir=output_dir,
        num_shards=num_shards,
    )


def load_prepared_dataset(output_dir: str = OUTPUT_DIR):
    """Load the tokenised dataset from disk."""

    return load_tokenized_dataset(output_dir)


def split_prepared_dataset(
    dataset: DatasetDict,
    *,
    validation_size: Optional[int] = VALIDATION_SIZE,
):
    """Split the prepared dataset into train and evaluation subsets."""

    return split_dataset(dataset, validation_size=validation_size)
