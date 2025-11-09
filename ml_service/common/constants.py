"""Common constants reused across training and inference modules."""

from __future__ import annotations

from lib.constants import (
    MAX_TRAINING_TIME_SECONDS,
    MAX_LENGTH,
    INPUT_IDS,
    ATTENTION_MASK,
    LABELS,
    TOKENIZER_NAME,
    OUTPUT_DIR,
    NUM_SHARDS,
    VALIDATION_SIZE,
    LR_MIN,
    LR_MAX,
)

__all__ = [
    "MAX_TRAINING_TIME_SECONDS",
    "MAX_LENGTH",
    "INPUT_IDS",
    "ATTENTION_MASK",
    "LABELS",
    "TOKENIZER_NAME",
    "OUTPUT_DIR",
    "NUM_SHARDS",
    "VALIDATION_SIZE",
    "LR_MIN",
    "LR_MAX",
]
