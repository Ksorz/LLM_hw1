"""Factory helpers to create a FastAPI application."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from fastapi import Depends, FastAPI, HTTPException

from .dependencies import AppDependencies
from .schemas import (
    BatchTextRequest,
    BatchTextResponse,
    MetadataResponse,
    TextRequest,
    TextResponse,
)

__all__ = ["create_app"]


def _default_predict(_: str) -> str:
    raise HTTPException(status_code=403, detail="модель не смогла обработать данные")


def _default_predict_batch(texts: Iterable[str]) -> List[str]:
    return [_default_predict(text) for text in texts]


def _default_metadata() -> Dict[str, str]:
    return {}


def create_app(deps: Optional[AppDependencies] = None) -> FastAPI:
    """Create the FastAPI application with injectable dependencies."""

    dependencies = deps or AppDependencies(
        predict=_default_predict,
        predict_batch=_default_predict_batch,
        metadata=_default_metadata,
    )

    app = FastAPI(title="LLM HW1 Service", version="1.0.0")

    @app.post("/forward", response_model=TextResponse)
    def forward(request: TextRequest, deps: AppDependencies = Depends(lambda: dependencies)) -> TextResponse:
        try:
            prediction = deps.predict(request.text)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime errors
            raise HTTPException(status_code=403, detail="модель не смогла обработать данные") from exc
        return TextResponse(prediction=prediction)

    @app.post("/forward_batch", response_model=BatchTextResponse)
    def forward_batch(
        request: BatchTextRequest, deps: AppDependencies = Depends(lambda: dependencies)
    ) -> BatchTextResponse:
        try:
            predictions = deps.predict_batch(request.texts)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime errors
            raise HTTPException(status_code=403, detail="модель не смогла обработать данные") from exc
        return BatchTextResponse(predictions=predictions)

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(deps: AppDependencies = Depends(lambda: dependencies)) -> MetadataResponse:
        data = deps.metadata()
        return MetadataResponse(**data)

    return app
