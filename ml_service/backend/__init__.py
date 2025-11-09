"""FastAPI backend package."""

from .api import create_app
from .dependencies import AppDependencies
from .schemas import (
    BatchTextRequest,
    BatchTextResponse,
    MetadataResponse,
    TextRequest,
    TextResponse,
)

__all__ = [
    "AppDependencies",
    "BatchTextRequest",
    "BatchTextResponse",
    "MetadataResponse",
    "TextRequest",
    "TextResponse",
    "create_app",
]
