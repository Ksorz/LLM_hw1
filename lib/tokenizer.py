"""Подготовка токенайзера."""

from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .constants import MAX_LENGTH, TOKENIZER_NAME


def build_tokenizer(
    name: str = TOKENIZER_NAME,
    *,
    use_fast: bool = True,
    padding_side: Optional[str] = "right",
) -> PreTrainedTokenizerBase:
    """Создать токенайзер и гарантировать наличие pad_token.

    Parameters
    ----------
    name:
        Имя модели токенайзера в HuggingFace Hub.
    use_fast:
        Использовать ли rust-реализацию.
    padding_side:
        Предпочитаемая сторона паддинга.
    """

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    if padding_side is not None:
        tokenizer.padding_side = padding_side

    if getattr(tokenizer, "model_max_length", None) and tokenizer.model_max_length < MAX_LENGTH:
        tokenizer.model_max_length = MAX_LENGTH

    return tokenizer


__all__ = ["build_tokenizer"]


