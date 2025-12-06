.PHONY: help serve serve-dev serve-trained test train-baseline docker-build docker-up clean up-full

help:
	@echo "Available commands:"
	@echo "  make serve             - Run API with baseline model"
	@echo "  make serve-dev         - Run API in dev mode (autoreload)"
	@echo "  make serve-trained     - Run API with trained model (requires CHECKPOINT)"
	@echo "  make test              - Run tests"
	@echo "  make train-baseline    - Run simple baseline training"
	@echo "  make docker-build      - Build all Docker images"
	@echo "  make docker-up         - Run API via Docker Compose"
	@echo "  make up-full           - Run API + DB + Frontend + Monitoring via Docker Compose"
	@echo "  make clean             - Remove artifacts"

serve:
	python serve.py

serve-dev:
	python serve.py --reload

serve-trained:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: CHECKPOINT variable is not set"; exit 1; fi
	python serve.py --checkpoint $(CHECKPOINT)

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

clean:
	rm -rf output_dir/
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
