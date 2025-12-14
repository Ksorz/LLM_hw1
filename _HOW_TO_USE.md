# Как использовать ML-сервис

## 🏗️_0: полный стек (Docker Compose)

Все сервисы (API, БД, фронтенд, мониторинг) поднимаются через Docker Compose.

Быстрый старт (PyTorch без чекпоинта, baseline):
```bash
make up-full
```

Полезные варианты:
- **API на PyTorch чекпоинте**:  
  `make up-full-ckpt` (возьмётся последний `checkpoint-*` или `CHECKPOINT=...`)
- **up-full-dev** — поднимает все сервисы, но API не стартует (`tail -f /dev/null`). Далее в контейнере `api` запускаем:
  - `make serve-ckpt` (GPU PyTorch - последний чекпоинт)
  - `make serve-base` (GPU PyTorch - необученная модель)

Что поднимается:
1. **Frontend (Streamlit)**: http://localhost:8501 — чат с моделью.
2. **API (FastAPI)**: http://localhost:8000/docs — Swagger UI.
3. **Grafana**: http://localhost:3000 — дашборды (login: `admin`/`admin`).
4. **Prometheus**: http://localhost:9090 — метрики.
5. **PostgreSQL**: БД (порт 5432).


### Запуск только API (локально, без Docker)

Если вы хотите запустить API без Docker (например, для отладки):

```bash
# Создать виртуальное окружение и поставить зависимости
make venv
source .venv/bin/activate

# (опц.) задать DATABASE_URL, если не используете docker db
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_service

# Запуск (baseline) или другие варианты:
make serve-base
make serve-onnx
make serve-ckpt
```

### Дополнительно

Команды `*-onnx` и `*-ckpt` можно указать с переменными: `ONNX=<path to model.onnx>` или `CHECKPOINT=<path to checkpoint folder>` соответственно. Если эти переменные не указаны, то make возьмёт самый свежий артефакт:
- `*-onnx`: последний model.onnx в output_dir/onnx
- `*-ckpt`: последний checkpoint-* в output_dir/gpt2-1b-russian
Аналогично, up-full-onnx / up-full-ckpt тоже подберут самый свежий, если не указать переменные.

---

### Интерфейсы

#### 🖥️ Frontend (Чат)
Откройте http://localhost:8501.
Здесь можно общаться с моделью в режиме диалога. История сообщений сохраняется в рамках сессии браузера.

#### 🔌 API Endpoints
- **POST /forward**: Генерация текста (логируется в БД).
- **POST /forward_batch**: Батчевая генерация.
- **GET /metadata**: Информация о модели.
- **GET /metrics**: Метрики для Prometheus.

#### 📊 Мониторинг
- **Grafana** (http://localhost:3000): Визуализация метрик. Вы можете добавить Prometheus как Data Source (`http://prometheus:9090`) и создать дашборд.
- **Prometheus** (http://localhost:9090): Сырые метрики. Попробуйте запросить `http_requests_total`.

#### 🗄️ База данных
Все запросы к API логируются в таблицу `request_logs` в PostgreSQL.
Поля: `timestamp`, `input_text`, `output_text`, `processing_time_ms`, `model_name`.

---

## 🏗️_1: ONNX и метаданные

Цель: экспортировать модель в ONNX, зашить мета (commit, date, experiment), уметь поднять сервис на ONNX и видеть мету в `/metadata`.

### Экспорт в ONNX (самодостаточная папка)
```bash
# Авто: последний checkpoint-* -> новый expN в output_dir/onnx/expN/model.onnx
make export-onnx

# Явно задать пути
make export-onnx \
  CHECKPOINT=/app/output_dir/gpt2-1b-russian/checkpoint-10000 \
  ONNX_OUT=/app/output_dir/onnx/exp9/model.onnx \
  EXPERIMENT=exp9
```
- В папке expN будут: model.onnx, model.onnx_data (external weights), tokenizer/config файлы, метаданные внутри ONNX.
- Имя эксперимента по умолчанию = имя чекпоинта (e.g. checkpoint-10000), если не указано вручную.
- Артефакт полностью самодостаточен: токенайзер и config лежат рядом с ONNX.

### Запуск на ONNX
- Локально: `make serve-onnx` (без аргументов возьмёт последний model.onnx), или `make serve-onnx ONNX=/app/output_dir/onnx/expN/model.onnx`
- Docker стек: `make up-full-onnx` (без аргументов возьмёт последний model.onnx), или `ONNX=... make up-full-onnx`
- `/metadata` берёт commit/date/experiment из ONNX; при отсутствии — fallback на env.


## 🏗️_2: DVC pipeline (extract → train → export ONNX)

Цель: собрать воспроизводимый pipeline с артефактами (шарды, чекпоинт, ONNX) и интегрировать метаданные в ONNX.

### Команды
- Полный прогон всех стадий:
```bash
make dvc-repro-all   # dvc repro export_onnx (extract_data -> train_model -> export_onnx)
```
- Только базовые стадии (extract -> train):
```bash
make dvc-repro       # dvc repro
```

### Стадии (dvc.yaml)
- `extract_data`: готовит parquet-шарды (`output_dir/dataset/`) по параметрам из `params.yaml`.
- `train_model`: обучает baseline, сохраняет чекпоинт в `output_dir/gpt2-1b-russian/checkpoint-*`.
- `export_onnx`: конвертирует чекпоинт в самодостаточный ONNX артефакт (`output_dir/onnx/expN/model.onnx` + tokenizer/config + metadata).

### Параметры
- Редактировать в `params.yaml`:
  - `extract_data`: датасет, split, max_length, num_shards, num_proc, output_dir
  - `train_model`: режим, batch_size, grad_accum, max_steps, lr, bf16, torch_compile, data_dir
  - `export_onnx`: checkpoint, onnx_out, experiment (опциональны, есть авто-подбор)

### Авто-подбор артефактов
- Если `CHECKPOINT` не задан: берётся последний `checkpoint-*` из `output_dir/gpt2-1b-russian`.
- Если `ONNX_OUT` не задан: создаётся новый `output_dir/onnx/expN/model.onnx` (N = следующий номер).
- `EXPERIMENT` по умолчанию = имя чекпоинта (например, `checkpoint-10000`), если не указано.
- Папка expN самодостаточна: `model.onnx`, `model.onnx_data`, tokenizer/config, метаданные внутри ONNX.

### Проверка результата
1) После `make dvc-repro-all` проверьте, что:
   - есть `output_dir/dataset/` (шарды);
   - есть `output_dir/gpt2-1b-russian/checkpoint-*`;
   - есть `output_dir/onnx/expN/model.onnx` и `model.onnx_data` + tokenizer/config.
2) Проверьте метаданные в ONNX:
```bash
python - <<'PY'
from ml_service.inference.service import read_onnx_metadata
print(read_onnx_metadata("/app/output_dir/onnx/expN/model.onnx"))
PY
```
3) Поднимите API на этом ONNX:
```bash
make serve-onnx ONNX=/app/output_dir/onnx/expN/model.onnx
# или docker-стек: make up-full-onnx ONNX=/app/output_dir/onnx/expN/model.onnx
```
4) Откройте `/metadata` — commit/date/experiment должны читаться из ONNX.

### Что дальше (для полного закрытия усложнения 2)
- Добавить DVC `metrics:`/`plots:` (loss/ppl) из тренировки или пост-обработки логов.
- (Опц.) Подключить DVC remote для хранения артефактов/метрик.
- (Опц.) Экспорт TensorBoard/W&B графиков в DVC plots или задокументировать W&B как основной трекер.


---
---
---
---
---
---

## End-to-End: Обучение → API

### Шаг 1: Обучение модели (как раньше)

```bash
make train-baseline
```

### Шаг 2: Запуск с обученной моделью

Если вы используете Docker Compose, вам нужно прокинуть путь к checkpoint в переменную окружения или `.env` файл, либо просто смонтировать volume.
Для простоты, можно запустить локально:

```bash
make serve-trained CHECKPOINT=/app/output_dir/gpt2-1b-russian/checkpoint-1000
```

### Шаг 2b: Экспорт в ONNX и запуск на ONNXRuntime 🏗️_1

```bash
# Экспорт чекпоинта в ONNX с метаданными
make export-onnx CHECKPOINT=/app/output_dir/gpt2-1b-russian/checkpoint-1000 ONNX_OUT=/app/output_dir/onnx/exp1/model.onnx EXPERIMENT=exp1

# Запуск сервиса на ONNX (локально)
make serve-onnx ONNX=/app/output_dir/onnx/exp1/model.onnx

# Запуск сервиса на PyTorch (baseline) 🏰
# make serve
```

Метаданные `/metadata` при работе через ONNX читаются прямо из модели (commit/date/experiment), при их отсутствии — из переменных окружения.

#### Экспорт через Optimum (для моделей, не поддержанных transformers.onnx) 🏗️_1
```bash
make export-onnx-optimum \
  CHECKPOINT=/app/output_dir/gpt2-1b-russian/checkpoint-1000 \
  ONNX_OUT=/app/output_dir/onnx/exp1 \
  EXPERIMENT=exp1
```
`ONNX_OUT` указывается как папка, внутри появится `model.onnx` и вспомогательные файлы.

#### Запуск стека с ONNX через Docker Compose 🏗️_1
1) Подготовьте ONNX-модель (см. экспорт выше) и убедитесь, что путь доступен внутри контейнера `api` (том или образ).  
2) В `docker-compose.yml` замените команду для `api` на:
```yaml
command: python serve.py --onnx /app/output_dir/onnx/exp1/model.onnx
```
или запустите разово так:
```bash
docker compose run -d --service-ports \
  -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  -e WANDB_API_KEY=${WANDB_API_KEY:-} \
  -e WANDB_PROJECT=${WANDB_PROJECT:-llm_service} \
  -e DATABASE_URL=postgresql://postgres:postgres@db:5432/llm_service \
  api python serve.py --onnx /app/output_dir/onnx/exp1/model.onnx
```

---

## 🏗️_3 Оценка модели: POST /evaluate (perplexity)

- Режим: только PyTorch запуск (`serve.py` без `--onnx`)
- Вход:
  - JSON: `{"texts": ["hello", "world"]}`
  - CSV (multipart): колонка `text`
- Выход: `{"count": N, "avg_loss": ..., "perplexity": ..., "duration_ms": ...}`

Пример (JSON):
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello", "world"]}'
```

Пример (CSV):
```bash
curl -X POST http://localhost:8000/evaluate \
  -F "file=@texts.csv"
# в texts.csv должна быть колонка 'text'
```

Если запущено в ONNX-режиме, эндпоинт вернёт 503 (не поддерживается).

---

## 🏗️_4 Добавление данных, переобучение и деплой

### Добавить данные
- Эндпоинт: `PUT /add_data`
- Вход: multipart CSV с колонкой `text` (обязательно) и `label` (опционально)
```bash
curl -X PUT http://localhost:8000/add_data \
  -F "file=@data.csv"
# data.csv: text,label
```
- Ответ: `{"inserted": N}`

### Переобучение (фоново, subprocess)
- Эндпоинт: `PUT /retrain`
- Вход (JSON): `{"gpu_id": "0", "checkpoint_path": "/app/output_dir/gpt2-1b-russian"}`
```bash
curl -X PUT http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"gpu_id":"0"}'
```
- Ответ: `{"experiment_id": ..., "status": "pending"}`

### Метрики эксперимента
- Эндпоинт: `GET /metrics/{experiment_id}`
- Ответ: `status`, `checkpoint_path`, `onnx_path`, `metrics` (если есть)

### Деплой (hot-reload модели)
- Эндпоинт: `POST /deploy/{experiment_id}`
- Требование: experiment в статусе `succeeded`.
- Если указан `onnx_path` — поднимется ONNXRuntime, иначе PyTorch checkpoint. Зависимости обновляются в процессе API (без рестарта контейнера).
- Ответ: `{"experiment_id": ..., "status": "deployed"}`

---

## 🏗️_5 Мониторинг и Grafana

- Prometheus собирает метрики с `api:8000/metrics` (см. `monitoring/prometheus.yml`).
- Grafana: http://localhost:3000 (admin/admin). Data Source: Prometheus `http://prometheus:9090`.
- Готовый дашборд: `monitoring/grafana-dashboard.json` (импортируйте в Grafana).
  - HTTP request rate по handler/method/status
  - Latency p50/p95
  - Error rate 4xx/5xx

---

## 🏗️_2 DVC pipeline (extract → train → export ONNX)

```bash
# Выполнить все стадии по dvc.yaml
make dvc-repro-all

# Только стандартные цели (по умолчанию = dvc repro)
make dvc-repro
```

Стадии:
- `extract_data` → подготавливает parquet-шарды (`output_dir/`)
- `train_model` → обучает baseline (параметры в `params.yaml`)
- `export_onnx` → экспортирует чекпоинт в ONNX (`output_dir/onnx/...`)

Параметры править в `params.yaml` (dataset, batch_size, max_steps и т.д.).

---

## Docker команды

### Управление контейнерами

```bash
# Запустить всё (с мониторингом) 🏗️_0
make up-full

# Запустить только API и БД
docker-compose up -d api db

# Пересобрать контейнеры (если изменили код) 🏗️_0
make docker-build

# Остановить всё
docker-compose down

# Посмотреть логи API
docker-compose logs -f api
```

### Переменные окружения (.env)

Создайте файл `.env` в корне проекта для настройки:

```env
CUDA_VISIBLE_DEVICES=0
DATABASE_URL=postgresql://postgres:postgres@db:5432/llm_service
GRAFANA_PASSWORD=admin
```

---

## Troubleshooting

### Ошибка: "Connection refused" к БД
Если API падает с ошибкой подключения к БД, подождите пару секунд. Docker Compose настроен на `healthcheck`, но иногда первый запуск Postgres занимает время. Контейнер `api` должен автоматически перезапуститься.

### Ошибка: "CUDA out of memory"
Уменьшите размер батча или `max_new_tokens` в запросе. Или запустите API на CPU:
`python serve.py --device cpu`

### Как подключиться к БД вручную?
```bash
docker exec -it llm_db psql -U postgres -d llm_service
# \dt - показать таблицы
# select * from request_logs limit 5; - посмотреть логи
```

---

## Дополнительная информация
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
