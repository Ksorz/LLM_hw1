"""Factory helpers to create a FastAPI application."""

from __future__ import annotations

import time
import io
import csv
import base64
from typing import Dict, Iterable, List, Optional
import os

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
import threading
import subprocess
from datetime import datetime
from lib.inference import load_model_for_inference
from ml_service.inference.service import OnnxTextGenerator, read_onnx_metadata, PerplexityCalculator

from .database import get_db, SessionLocal
from .dependencies import AppDependencies
from .models import RequestLog, DatasetSample, Experiment
from .schemas import (
    BatchTextRequest,
    BatchTextResponse,
    MetadataResponse,
    TextRequest,
    TextResponse,
    EvaluateResponse,
    AddDataResponse,
    RetrainRequest,
    RetrainResponse,
    MetricsResponse,
    DeployResponse,
    ExperimentStatus,
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


def _update_experiment_status(db: Session, exp_id: int, **fields):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if exp is None:
        return
    for k, v in fields.items():
        setattr(exp, k, v)
    db.commit()
    db.refresh(exp)


def create_app(deps: Optional[AppDependencies] = None) -> FastAPI:
    """Create the FastAPI application with injectable dependencies."""

    dependencies = deps or AppDependencies(
        predict=_default_predict,
        predict_batch=_default_predict_batch,
        metadata=_default_metadata,
        evaluate_perplexity=None,
    )
    deps_ref = {"deps": dependencies}
    deps_lock = threading.Lock()
    experiment_lock = threading.Lock()

    app = FastAPI(title="LLM HW1 Service", version="1.0.0")

    def _run_retrain_job(exp_id: int, gpu_id: Optional[str], checkpoint_path: Optional[str]):
        db = SessionLocal()
        try:
            _update_experiment_status(
                db,
                exp_id,
                status=ExperimentStatus.running.value,
                started_at=datetime.utcnow(),
            )

            env = os.environ.copy()
            if gpu_id:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

            cmd = ["python", "train_distributed.py", "--mode", "baseline"]
            if checkpoint_path:
                # train_distributed не принимает прямой output путь,
                # но можем оставить checkpoint_path для метаданных.
                pass

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            status = (
                ExperimentStatus.succeeded.value
                if result.returncode == 0
                else ExperimentStatus.failed.value
            )
            metrics_payload = None
            final_ckpt = checkpoint_path or "output_dir/gpt2-1b-russian"

            _update_experiment_status(
                db,
                exp_id,
                status=status,
                finished_at=datetime.utcnow(),
                checkpoint_path=final_ckpt,
                metrics=metrics_payload,
            )
        except Exception:
            _update_experiment_status(
                db,
                exp_id,
                status=ExperimentStatus.failed.value,
                finished_at=datetime.utcnow(),
            )
        finally:
            db.close()

    @app.post("/forward", response_model=TextResponse)
    async def forward(
        request: Request,
        background_tasks: BackgroundTasks,
        image: Optional[UploadFile] = File(None),
        deps: AppDependencies = Depends(lambda: deps_ref["deps"]),
        db: Session = Depends(get_db),
    ) -> TextResponse | PlainTextResponse:
        # Требование ТЗ: /forward принимает либо JSON (без изображений),
        # либо multipart/form-data с image.
        # Если формат неверный — вернуть 400 plain-text "bad request".
        #
        # Для multipart: дополнительные параметры берём из headers (минимально поддерживаем X-Text).
        validated_text: Optional[str] = None
        image_b64: Optional[str] = None

        if image is not None:
            header_text = (
                request.headers.get("x-text")
                or request.headers.get("x_prompt")
                or request.headers.get("x-prompt")
            )
            if not header_text or not header_text.strip():
                return PlainTextResponse("bad request", status_code=400)

            try:
                raw = await image.read()
            except Exception:
                return PlainTextResponse("bad request", status_code=400)

            if not raw:
                return PlainTextResponse("bad request", status_code=400)

            validated_text = header_text.strip()
            image_b64 = base64.b64encode(raw).decode("ascii")
        else:
            try:
                payload = await request.json()
                validated = TextRequest(**payload)
                validated_text = validated.text
            except Exception:
                return PlainTextResponse("bad request", status_code=400)

        start_time = time.time()
        try:
            prediction = deps.predict(validated_text or "")
        except HTTPException:
            raise
        except Exception:  # pragma: no cover - runtime errors
            return PlainTextResponse("модель не смогла обработать данные", status_code=403)

        duration = time.time() - start_time

        # Log metadata if available to get model name/device
        meta = deps.metadata()
        model_name = meta.get("experiment") or meta.get("checkpoint")
        device = meta.get("device")

        background_tasks.add_task(
            log_request,
            db,
            validated_text or "",
            prediction,
            duration,
            model_name,
            device
        )

        return TextResponse(prediction=prediction, image_base64=image_b64)

    @app.post("/forward_batch", response_model=BatchTextResponse)
    def forward_batch(
        request: BatchTextRequest,
        background_tasks: BackgroundTasks,
        deps: AppDependencies = Depends(lambda: deps_ref["deps"]),
        db: Session = Depends(get_db),
    ) -> BatchTextResponse:
        start_time = time.time()
        try:
            predictions = deps.predict_batch(request.texts)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime errors
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные",
            ) from exc

        duration = time.time() - start_time

        # Log metadata
        meta = deps.metadata()
        model_name = meta.get("experiment") or meta.get("checkpoint")
        device = meta.get("device")

        # Log each item in the batch
        # Note: simplistic time division. Better to measure per item if possible,
        # but batch processing is usually monolithic.
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
    def metadata(deps: AppDependencies = Depends(lambda: deps_ref["deps"])) -> MetadataResponse:
        data = deps.metadata()
        return MetadataResponse(**data)

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate(
        request: Request,
        file: Optional[UploadFile] = File(None),
        deps: AppDependencies = Depends(lambda: deps_ref["deps"]),
    ) -> EvaluateResponse:
        if deps.evaluate_perplexity is None:
            raise HTTPException(status_code=503, detail="evaluate not supported for this model")

        texts: List[str] = []
        if file is not None:
            try:
                content = await file.read()
                decoded = content.decode("utf-8")
                reader = csv.DictReader(io.StringIO(decoded))
                if not reader.fieldnames or "text" not in reader.fieldnames:
                    raise HTTPException(status_code=400, detail="CSV must contain 'text' column")
                for row in reader:
                    val = row.get("text", "")
                    if val.strip():
                        texts.append(val)
            except HTTPException:
                raise
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=400, detail=f"failed to parse CSV: {exc}") from exc
        else:
            try:
                payload = await request.json()
                texts = payload.get("texts", [])
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

        if not texts:
            raise HTTPException(status_code=422, detail="texts must not be empty")

        start = time.time()
        metrics = deps.evaluate_perplexity(texts)
        duration_ms = (time.time() - start) * 1000

        return EvaluateResponse(
            count=int(metrics.get("count", len(texts))),
            avg_loss=float(metrics["avg_loss"]),
            perplexity=float(metrics["perplexity"]),
            duration_ms=duration_ms,
        )

    @app.put("/add_data", response_model=AddDataResponse)
    async def add_data(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ) -> AddDataResponse:
        try:
            content = await file.read()
            decoded = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(decoded))
            if not reader.fieldnames or "text" not in reader.fieldnames:
                raise HTTPException(status_code=400, detail="CSV must contain 'text' column")
            rows = []
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                label = (row.get("label") or "").strip() or None
                rows.append(DatasetSample(text=text, label=label))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to parse CSV: {exc}") from exc

        if not rows:
            raise HTTPException(status_code=422, detail="no valid rows found")

        db.bulk_save_objects(rows)
        db.commit()
        return AddDataResponse(inserted=len(rows))

    @app.put("/retrain", response_model=RetrainResponse)
    def retrain(
        request: RetrainRequest,
        db: Session = Depends(get_db),
    ) -> RetrainResponse:
        exp = Experiment(
            status=ExperimentStatus.pending.value,
            checkpoint_path=request.checkpoint_path,
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)

        with experiment_lock:
            thread = threading.Thread(
                target=_run_retrain_job,
                args=(exp.id, request.gpu_id, request.checkpoint_path),
                daemon=True,
            )
            thread.start()

        return RetrainResponse(experiment_id=exp.id, status=ExperimentStatus.pending)

    @app.get("/metrics/{experiment_id}", response_model=MetricsResponse)
    def metrics_endpoint(
        experiment_id: int,
        db: Session = Depends(get_db),
    ) -> MetricsResponse:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return MetricsResponse(
            experiment_id=exp.id,
            status=ExperimentStatus(exp.status),
            checkpoint_path=exp.checkpoint_path,
            onnx_path=exp.onnx_path,
            metrics=exp.metrics,
        )

    @app.post("/deploy/{experiment_id}", response_model=DeployResponse)
    def deploy(
        experiment_id: int,
        db: Session = Depends(get_db),
    ) -> DeployResponse:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        if exp.status != ExperimentStatus.succeeded.value:
            raise HTTPException(status_code=400, detail="experiment not in succeeded state")

        if exp.onnx_path:
            generator = OnnxTextGenerator.from_checkpoint(
                exp.onnx_path,
                tokenizer=None,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            def predict(text: str) -> str:
                return generator.predict(text)

            def predict_batch(texts):
                return generator.predict_batch(texts)

            def metadata() -> Dict[str, str]:
                meta = read_onnx_metadata(exp.onnx_path)
                meta.setdefault("checkpoint", exp.onnx_path)
                meta.setdefault("experiment", os.path.basename(exp.onnx_path))
                meta.setdefault("commit", os.getenv("GIT_COMMIT", "unknown"))
                meta.setdefault("date", datetime.utcnow().isoformat())
                meta.setdefault("device", "onnxruntime")
                return meta

            new_deps = AppDependencies(
                predict=predict,
                predict_batch=predict_batch,
                metadata=metadata,
                evaluate_perplexity=None,
            )
        else:
            service = load_model_for_inference(
                checkpoint_path=exp.checkpoint_path,
                tokenizer_path=None,
                device=None,
                max_new_tokens=50,
            )
            calculator = PerplexityCalculator(
                model=service.model,
                tokenizer=service.tokenizer,
                device=service.device,
            )

            def predict(text: str) -> str:
                return service.predict(text)

            def predict_batch(texts):
                return service.predict_batch(texts)

            def metadata() -> Dict[str, str]:
                return {
                    "commit": os.getenv("GIT_COMMIT", "unknown"),
                    "date": datetime.utcnow().isoformat(),
                    "experiment": os.path.basename(exp.checkpoint_path or "none"),
                    "checkpoint": exp.checkpoint_path or "none",
                    "device": str(service.device),
                }

            def evaluate_perplexity(texts):
                return calculator.compute(list(texts))

            new_deps = AppDependencies(
                predict=predict,
                predict_batch=predict_batch,
                metadata=metadata,
                evaluate_perplexity=evaluate_perplexity,
            )

        with deps_lock:
            deps_ref["deps"] = new_deps

        return DeployResponse(experiment_id=experiment_id, status="deployed")

    @app.get("/health")
    def health() -> Dict[str, str]:
        # Проверка здоровья приложения
        return {"status": "ok"}

    return app
