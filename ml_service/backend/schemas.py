"""Pydantic models used by the FastAPI application."""

from __future__ import annotations

from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field, validator

__all__ = [
    "TextRequest",
    "TextResponse",
    "BatchTextRequest",
    "BatchTextResponse",
    "MetadataResponse",
    "EvaluateRequest",
    "EvaluateResponse",
    "AddDataResponse",
    "RetrainRequest",
    "RetrainResponse",
    "MetricsResponse",
    "DeployResponse",
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
    image_base64: Optional[str] = None


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


class EvaluateRequest(BaseModel):
    texts: List[str]

    @validator("texts")
    def _check_texts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("texts must not be empty")
        return value


class EvaluateResponse(BaseModel):
    count: int
    avg_loss: float
    perplexity: float
    duration_ms: float


class AddDataResponse(BaseModel):
    inserted: int


class ExperimentStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class RetrainRequest(BaseModel):
    gpu_id: Optional[str] = None
    checkpoint_path: Optional[str] = None


class RetrainResponse(BaseModel):
    experiment_id: int
    status: ExperimentStatus


class MetricsResponse(BaseModel):
    experiment_id: int
    status: ExperimentStatus
    checkpoint_path: Optional[str] = None
    onnx_path: Optional[str] = None
    metrics: Optional[dict] = None


class DeployResponse(BaseModel):
    experiment_id: int
    status: str = "deployed"
