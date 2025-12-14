.PHONY: help serve-base serve-ckpt serve-onnx test train-baseline export-onnx docker-build docker-up clean up-full dvc-repro dvc-repro-extract dvc-repro-train dvc-repro-export venv
.PHONY: up-full-dev
.PHONY: up-full-onnx up-full-ckpt

# Авто-поиск последних артефактов (если переменные не заданы)
LATEST_ONNX      := $(shell find output_dir/onnx -type f -name 'model.onnx' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $$2}')
LATEST_CKPT      := $(shell find output_dir/gpt2-1b-russian -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $$2}')
ONNX_PATH        ?= $(if $(ONNX),$(ONNX),$(LATEST_ONNX))
CHECKPOINT_PATH  ?= $(if $(CHECKPOINT),$(CHECKPOINT),$(LATEST_CKPT))

help:
	@echo "Available commands:"
	@echo "  make serve-base        - Run API with baseline model"
	@echo "  make serve-ckpt        - Run API with trained model (auto-picks latest if CHECKPOINT not set)"
	@echo "  make serve-onnx        - Run API with ONNX model (auto-picks latest if ONNX not set)"
	@echo "  make test              - Run tests"
	@echo "  make train-baseline    - Run simple baseline training"
	@echo "  make export-onnx       - Export checkpoint to ONNX (requires CHECKPOINT, ONNX_OUT)"
	@echo "  make docker-build      - Build all Docker images"
	@echo "  make docker-up         - Run API via Docker Compose"
	@echo "  make up-full           - Run API + DB + Frontend + Monitoring via Docker Compose"
	@echo "  make up-full-onnx      - Run full stack with API on ONNX model (ONNX=...)"
	@echo "  make up-full-ckpt      - Run full stack with API on PyTorch checkpoint (CHECKPOINT=...)"
	@echo "  make dvc-repro         - Run DVC pipeline (default targets)"
	@echo "  make dvc-repro-extract - Run DVC stage extract_data"
	@echo "  make dvc-repro-train   - Run DVC stage train_model"
	@echo "  make dvc-repro-export  - Run DVC stage export_onnx"
	@echo "  make run-api-base      - Run API (inside container shell): baseline"
	@echo "  make run-api-ckpt      - Run API (inside container shell): checkpoint (CHECKPOINT=...)"
	@echo "  make run-api-onnx      - Run API (inside container shell): ONNX (ONNX=...)"
	@echo "  make venv              - Create .venv and install requirements (local dev)"
	@echo "  make clean             - Remove artifacts"

serve-base:
	python serve.py

serve-ckpt:
	@if [ -z "$(CHECKPOINT_PATH)" ]; then echo "Error: CHECKPOINT not set and no checkpoint-* found in output_dir/gpt2-1b-russian"; exit 1; fi
	@echo "Using checkpoint: $(CHECKPOINT_PATH)"
	python serve.py --checkpoint "$(CHECKPOINT_PATH)"

serve-onnx:
	@if [ -z "$(ONNX_PATH)" ]; then echo "Error: ONNX not set and no model.onnx found under output_dir/onnx"; exit 1; fi
	@echo "Using ONNX: $(ONNX_PATH)"
	python serve.py --onnx "$(ONNX_PATH)"

test:
	pytest tests/

train-baseline:
	CUDA_VISIBLE_DEVICES=0 python train_distributed.py \
		--mode baseline \
		--bf16 \
		--batch-size 8 \
		--grad-accum 4 \
		--run-name test_baseline \
		--max-steps 100

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d api

up-full:
	docker-compose --profile monitoring up -d

# Поднять все сервисы, но API оставить в спящем режиме (запускать вручную через exec).
up-full-dev:
	API_CMD="tail -f /dev/null" docker-compose --profile monitoring up -d

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

up-full-onnx:
	@if [ -z "$(ONNX_PATH)" ]; then echo "Error: ONNX not set and no model.onnx found under output_dir/onnx"; exit 1; fi
	@echo "Using ONNX: $(ONNX_PATH)"
	API_CMD="python serve.py --onnx $(ONNX_PATH)" docker-compose --profile monitoring up -d

up-full-ckpt:
	@if [ -z "$(CHECKPOINT_PATH)" ]; then echo "Error: CHECKPOINT not set and no checkpoint-* found in output_dir/gpt2-1b-russian"; exit 1; fi
	@echo "Using checkpoint: $(CHECKPOINT_PATH)"
	API_CMD="python serve.py --checkpoint $(CHECKPOINT_PATH)" docker-compose --profile monitoring up -d

clean:
	rm -rf output_dir/
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

export-onnx:
	python export_onnx.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) $(if $(ONNX_OUT),--output $(ONNX_OUT),) $(if $(EXPERIMENT),--experiment $(EXPERIMENT),)

dvc-repro:
	CUDA_VISIBLE_DEVICES=1 dvc repro

dvc-repro-extract:
	CUDA_VISIBLE_DEVICES=1 dvc repro extract_data

dvc-repro-train:
	CUDA_VISIBLE_DEVICES=1 dvc repro train_model

dvc-repro-export:
	CUDA_VISIBLE_DEVICES=1 dvc repro export_onnx
