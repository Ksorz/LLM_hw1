.PHONY: help serve serve-dev serve-trained test clean install

# Цвета для вывода
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Показать справку
	@echo "$(BLUE)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить зависимости
	@echo "$(BLUE)Установка зависимостей...$(NC)"
	pip install -r requirements.txt

serve: ## Запустить сервис с необученной моделью (baseline)
	@echo "$(BLUE)Запуск сервиса с необученной моделью...$(NC)"
	python serve.py --host 0.0.0.0 --port 8000

serve-dev: ## Запустить сервис в режиме разработки (с auto-reload)
	@echo "$(BLUE)Запуск сервиса в режиме разработки...$(NC)"
	python serve.py --host 0.0.0.0 --port 8000 --reload

serve-trained: ## Запустить сервис с обученной моделью (требуется CHECKPOINT)
	@echo "$(BLUE)Запуск сервиса с обученной моделью...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(YELLOW)Ошибка: укажите CHECKPOINT=<path>$(NC)"; \
		exit 1; \
	fi
	python serve.py --checkpoint $(CHECKPOINT) --host 0.0.0.0 --port 8000

test: ## Запустить тесты
	@echo "$(BLUE)Запуск тестов...$(NC)"
	python test_api.py

test-all: ## Запустить все тесты (включая pytest)
	@echo "$(BLUE)Запуск всех тестов...$(NC)"
	pytest tests/ -v

train-baseline: ## Запустить обучение baseline модели
	@echo "$(BLUE)Запуск обучения baseline...$(NC)"
	CUDA_VISIBLE_DEVICES=0 python train_distributed.py \
		--mode baseline \
		--bf16 \
		--batch-size 16 \
		--grad-accum 4 \
		--run-name baseline_test \
		--max-steps 100

docker-build: ## Собрать Docker образ
	@echo "$(BLUE)Сборка Docker образа...$(NC)"
	docker build -t llm_service:latest .

docker-up: ## Запустить Docker Compose
	@echo "$(BLUE)Запуск Docker Compose...$(NC)"
	docker-compose up -d

docker-down: ## Остановить Docker Compose
	@echo "$(BLUE)Остановка Docker Compose...$(NC)"
	docker-compose down

docker-logs: ## Показать логи Docker Compose
	docker-compose logs -f

clean: ## Очистить временные файлы
	@echo "$(BLUE)Очистка временных файлов...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

lint: ## Проверить код линтером
	@echo "$(BLUE)Проверка кода...$(NC)"
	ruff check lib/ ml_service/ --fix

format: ## Отформатировать код
	@echo "$(BLUE)Форматирование кода...$(NC)"
	black lib/ ml_service/ serve.py train_distributed.py

.DEFAULT_GOAL := help

