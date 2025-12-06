"""Factory helpers to create a FastAPI application."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from .database import get_db
from .dependencies import AppDependencies
from .models import RequestLog
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


def log_request(
    db: Session,
    input_text: str,
    output_text: str,
    processing_time: float,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
):
    """Log the request to the database."""
    try:
        log_entry = RequestLog(
            input_text=input_text,
            output_text=output_text,
            processing_time_ms=processing_time * 1000,
            model_name=model_name,
            device=device,
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        # Don't fail the request if logging fails
        pass


def create_app(deps: Optional[AppDependencies] = None) -> FastAPI:
    """Create the FastAPI application with injectable dependencies."""

    dependencies = deps or AppDependencies(
        predict=_default_predict,
        predict_batch=_default_predict_batch,
        metadata=_default_metadata,
    )

    app = FastAPI(title="LLM HW1 Service", version="1.0.0")

    @app.post("/forward", response_model=TextResponse)
    def forward(
        request: TextRequest,
        background_tasks: BackgroundTasks,
        deps: AppDependencies = Depends(lambda: dependencies),
        db: Session = Depends(get_db),
    ) -> TextResponse:
        start_time = time.time()
        try:
            prediction = deps.predict(request.text)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime errors
            raise HTTPException(status_code=403, detail="модель не смогла обработать данные") from exc
        
        duration = time.time() - start_time
        
        # Log metadata if available to get model name/device
        meta = deps.metadata()
        model_name = meta.get("experiment") or meta.get("checkpoint")
        device = meta.get("device")

        background_tasks.add_task(
            log_request,
            db,
            request.text,
            prediction,
            duration,
            model_name,
            device
        )
        
        return TextResponse(prediction=prediction)

    @app.post("/forward_batch", response_model=BatchTextResponse)
    def forward_batch(
        request: BatchTextRequest,
        background_tasks: BackgroundTasks,
        deps: AppDependencies = Depends(lambda: dependencies),
        db: Session = Depends(get_db),
    ) -> BatchTextResponse:
        start_time = time.time()
        try:
            predictions = deps.predict_batch(request.texts)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime errors
            raise HTTPException(status_code=403, detail="модель не смогла обработать данные") from exc
        
        duration = time.time() - start_time
        
        # Log metadata
        meta = deps.metadata()
        model_name = meta.get("experiment") or meta.get("checkpoint")
        device = meta.get("device")

        # Log each item in the batch
        # Note: simplistic time division, better to measure per item if possible but batch processing is usually monolithic
        avg_duration = duration / len(request.texts) if request.texts else 0
        
        for text, pred in zip(request.texts, predictions):
             background_tasks.add_task(
                log_request,
                db,
                text,
                pred,
                avg_duration,
                model_name,
                device
            )

        return BatchTextResponse(predictions=predictions)

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(deps: AppDependencies = Depends(lambda: dependencies)) -> MetadataResponse:
        data = deps.metadata()
        return MetadataResponse(**data)

    return app
