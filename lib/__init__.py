"""Высокоуровневый API проекта для обучения и подготовки данных."""

from .constants import (
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

from .config import (
    BASE_TRAINING_CONFIG,
    BEST_HP,
    SWEEP_CONFIG,
    build_training_config,
)

from .tokenizer import build_tokenizer
from .data import (
    prepare_raw_dataset,
    tokenize_dataset,
    save_dataset_shards,
    prepare_tokenized_dataset,
    load_tokenized_dataset,
    split_dataset,
)
from .modeling import create_model
from .optim import build_optimizer
from .schedulers import build_custom_scheduler, build_custom_scheduler_v2
from .callbacks import TimeoutCallback, InspectCallback
from .training import (
    build_training_arguments,
    create_trainer,
    run_training,
)
from .deepspeed import build_deepspeed_config
from .fsdp import build_fsdp_config
from .notebook_utils import (
    setup_multiprocessing_for_notebook,
    get_distributed_env_info,
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
    "BASE_TRAINING_CONFIG",
    "BEST_HP",
    "SWEEP_CONFIG",
    "build_training_config",
    "build_tokenizer",
    "prepare_raw_dataset",
    "tokenize_dataset",
    "save_dataset_shards",
    "prepare_tokenized_dataset",
    "load_tokenized_dataset",
    "split_dataset",
    "create_model",
    "build_optimizer",
    "build_custom_scheduler",
    "build_custom_scheduler_v2",
    "TimeoutCallback",
    "InspectCallback",
    "build_training_arguments",
    "create_trainer",
    "run_training",
    "build_deepspeed_config",
    "build_fsdp_config",
    "setup_multiprocessing_for_notebook",
    "get_distributed_env_info",
]

