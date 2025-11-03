"""Глобальные константы проекта."""

MAX_TRAINING_TIME_SECONDS = 60 * 30
MAX_LENGTH = 512

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"

TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "/app/output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

LR_MIN = 7e-5
LR_MAX = 7e-4


