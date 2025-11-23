# Как использовать ML-сервис

## Быстрый старт

### 1. Запуск сервиса с необученной моделью (baseline)

```bash
make serve
```

Или напрямую:

```bash
python serve.py
```

Сервис запустится на `http://0.0.0.0:8000`

### 2. Проверка работы API

Откройте Swagger UI: http://localhost:8000/docs

Или используйте curl:

```bash
# POST /forward - генерация текста
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{"text": "В начале было слово"}'

# POST /forward_batch - батчевая генерация
curl -X POST "http://localhost:8000/forward_batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Привет", "Как дела"]}'

# GET /metadata - метаданные модели
curl "http://localhost:8000/metadata"
```

---

## End-to-End: Обучение → Сохранение → Загрузка → API

### Шаг 1: Обучение модели

```bash
# Baseline (single GPU)
CUDA_VISIBLE_DEVICES=0 python train_distributed.py \
  --mode baseline \
  --bf16 \
  --batch-size 16 \
  --grad-accum 4 \
  --run-name my_experiment \
  --max-steps 1000

# DeepSpeed (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes 2 \
  --main_process_port 29500 \
  train_distributed.py \
    --mode deepspeed \
    --deepspeed-stage 2 \
    --bf16 \
    --batch-size 16 \
    --grad-accum 4 \
    --run-name my_deepspeed_experiment

# FSDP (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes 2 \
  train_distributed.py \
    --mode fsdp \
    --bf16 \
    --batch-size 16 \
    --run-name my_fsdp_experiment
```

### Шаг 2: Найти сохранённый checkpoint

После обучения модель сохраняется в `output_dir`:

```bash
ls -lh /app/output_dir/gpt2-1b-russian/
# Ищем папки вида checkpoint-XXXX
```

### Шаг 3: Запустить API с обученной моделью

```bash
# Через Makefile
make serve-trained CHECKPOINT=/app/output_dir/gpt2-1b-russian/checkpoint-1000

# Или напрямую
python serve.py --checkpoint /app/output_dir/gpt2-1b-russian/checkpoint-1000
```

### Шаг 4: Протестировать API

```bash
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{"text": "Машинное обучение это"}'
```

---

## Docker

### Локальный запуск (текущий контейнер)

Вы уже находитесь внутри Docker контейнера. Просто используйте `make serve`.

### Docker Compose (для production)

```bash
# Запустить только API
docker-compose up -d api

# Запустить с мониторингом (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Остановить
docker-compose down

# Посмотреть логи
docker-compose logs -f api
```

### Переменные окружения

Создайте `.env` файл:

```env
CUDA_VISIBLE_DEVICES=0
WANDB_API_KEY=your_key_here
WANDB_PROJECT=llm_service
GRAFANA_PASSWORD=admin
```

---

## Makefile команды

```bash
make help              # Показать все команды
make serve             # Запустить с необученной моделью
make serve-dev         # Запустить с auto-reload
make serve-trained     # Запустить с обученной моделью (CHECKPOINT=...)
make test              # Запустить базовые тесты
make train-baseline    # Быстрое обучение для теста
make docker-build      # Собрать Docker образ
make docker-up         # Запустить Docker Compose
make clean             # Очистить временные файлы
```

---

## Параметры serve.py

```bash
python serve.py --help

# Основные параметры:
--checkpoint PATH       # Путь к checkpoint
--device cuda/cpu       # Устройство (по умолчанию auto)
--max-new-tokens N      # Макс. токенов генерации (по умолчанию 50)
--host HOST             # Host (по умолчанию 0.0.0.0)
--port PORT             # Port (по умолчанию 8000)
--reload                # Auto-reload для разработки
```

---

## Примеры использования

### Python клиент

```python
import requests

# Генерация текста
response = requests.post(
    "http://localhost:8000/forward",
    json={"text": "Искусственный интеллект"}
)
print(response.json()["prediction"])

# Батчевая генерация
response = requests.post(
    "http://localhost:8000/forward_batch",
    json={"texts": ["Текст 1", "Текст 2", "Текст 3"]}
)
print(response.json()["predictions"])

# Метаданные
response = requests.get("http://localhost:8000/metadata")
print(response.json())
```

### JavaScript клиент

```javascript
// Генерация текста
fetch('http://localhost:8000/forward', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Привет, мир!'})
})
.then(res => res.json())
.then(data => console.log(data.prediction));
```

---

## Troubleshooting

### Ошибка: "CUDA out of memory"

```bash
# Уменьшите max-new-tokens
python serve.py --max-new-tokens 20

# Или используйте CPU
python serve.py --device cpu
```

### Ошибка: "Port 8000 already in use"

```bash
# Используйте другой порт
python serve.py --port 8001
```

### Модель генерирует бессмыслицу

Это нормально для необученной модели. Обучите модель или загрузите checkpoint.

---

## Мониторинг (опционально)

После запуска с `--profile monitoring`:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Дополнительная информация

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI схема**: http://localhost:8000/openapi.json

