# Current Project Progress

## Overview
This repository contains the refactored implementation of the HW1 language-model training project. The codebase has been reorganized into a dedicated `ml_service` package that exposes reusable building blocks for dataset preparation, model construction, training workflow orchestration, and the future inference service.

## Repository Structure
- `ml_service/`
  - `backend/` – FastAPI application scaffolding with request/response schemas and dependency injection hooks. The routes currently act as stubs that return a 403 error until the inference pipeline is connected.
  - `common/` – Shared constants and configuration helpers that consolidate default hyper-parameters and directories.
  - `data/` – Dataset preparation helpers that wrap the Hugging Face `datasets` API for sharding and train/validation splitting.
  - `inference/` – ONNX Runtime wrapper ready to host exported models once conversion is completed.
  - `training/` – High level workflow factory (`build_training_artifacts`) that bundles tokenizer, datasets, model, optimizer, scheduler, Trainer, and distributed configs into a single dataclass.
  - `pipeline/` & `utils/` – Reserved namespaces for the upcoming automation pipeline and shared utilities.
- `lib/` – Original homework modules with detailed implementations for tokenizer creation, dataset IO, schedulers, optimizer, training loops, distributed strategies, and logging utilities.
- `solution.py` – Compatibility entry point that forwards legacy helper calls to the new training workflow utilities.
- `data/`, `hw2_parallel_pretrain/`, `train_distributed.py`, etc. – Existing artifacts and scripts from the initial homework delivery.

## Current Status
- ✅ **Project restructuring:** Core code is now exposed through the `ml_service` package, enabling reuse between training scripts and the upcoming web service.
- ⚠️ **Inference service:** FastAPI routes and dependency injection are scaffolded but still wired to placeholder handlers that always return a 403 error. Real model loading, JSON/multipart parsing, and error handling must be implemented.
- ⚠️ **Training workflow:** `build_training_artifacts` assembles training components and is integrated with `solution.py`, but continuous training pipeline automation (DVC/MLflow) is not yet in place.
- ❌ **Model export & ONNX metadata:** Conversion of trained checkpoints to ONNX, metadata embedding (commit hash, save date, experiment name), and `/metadata` population are pending.
- ❌ **Docker Compose environment (Frontend, Backend, DB, Monitoring):** Containers, orchestration, and observability stack not started.
- ❌ **Evaluation & batch APIs:** `/forward_batch` currently returns stub data; `/evaluate` has not been implemented.
- ❌ **Continuous learning endpoints:** `/add_data`, `/retrain`, `/metrics/<id>`, and `/deploy/<id>` remain unimplemented.

## Next Steps Toward Minimum Requirements
1. Connect the inference service to a real model pipeline:
   - Load a tokenizer and trained checkpoint (PyTorch first, then ONNX runtime).
   - Implement JSON and multipart input validation, including optional header parameters for image-driven requests.
   - Return generated predictions and, where required, base64-encoded images.
2. Finalize error handling for 400 (bad request) and 403 (model failure) paths.
3. Provide functional `/forward_batch` behavior that leverages batched inference.

## Roadmap for Advanced Enhancements
1. **Uslovnie 0 – Docker Compose stack**
   - Build separate images for backend API, basic frontend UI, PostgreSQL feature store, and monitoring (Prometheus + Grafana).
   - Ensure backend exposes Prometheus metrics (e.g., via `prometheus-fastapi-instrumentator`).
2. **Uslovnie 1 – ONNX export & metadata endpoint**
   - Convert the trained PyTorch model to ONNX with dynamic axes.
   - Embed commit hash, export timestamp, and experiment name as ONNX metadata and expose them via `/metadata`.
3. **Uslovnie 2 – Training pipeline automation**
   - Create DVC (or MLflow) stages for `extract_data`, `train_model`, and `inference_model`.
   - Log metrics, graphs, and saved models; integrate TensorBoard for gradient-based runs.
4. **Uslovnie 3 – Batch evaluation APIs**
   - Implement `/forward_batch` for high-volume inference and `/evaluate` for CSV/ZIP dataset scoring with metric calculation.
5. **Uslovnie 4 – Continuous learning lifecycle**
   - Add `/add_data`, `/retrain`, `/metrics/<experiment_id>`, and `/deploy/<experiment_id>` endpoints.
   - Persist experiment metadata, metrics, and model artifacts; enable hot-swapping of active models.

This document should be updated as each milestone is completed.
