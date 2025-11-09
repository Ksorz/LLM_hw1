"""Pydantic models used by the FastAPI application."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator

__all__ = [
    "TextRequest",
    "TextResponse",
    "BatchTextRequest",
    "BatchTextResponse",
    "MetadataResponse",
]


class TextRequest(BaseModel):
    text: str = Field(..., description="The input text prompt")

    @validator("text")
    def _check_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class TextResponse(BaseModel):
    prediction: str


class BatchTextRequest(BaseModel):
    texts: List[str]

    @validator("texts")
    def _check_texts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("texts must not be empty")
        return value


class BatchTextResponse(BaseModel):
    predictions: List[str]


class MetadataResponse(BaseModel):
    commit: Optional[str] = None
    date: Optional[str] = None
    experiment: Optional[str] = None
