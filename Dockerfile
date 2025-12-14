FROM registry.k.avito.ru/nvcr-proxy/nvidia/pytorch:24.12-py3 AS base

ARG PROJECT_ROOT=/app
ENV PROJECT_ROOT=${PROJECT_ROOT}
WORKDIR $PROJECT_ROOT
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip

COPY requirements.txt $PROJECT_ROOT/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r $PROJECT_ROOT/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-build-isolation \
    'flash-attn==2.7.3'

ENV HF_DISABLE_TELEMETRY=true \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    SERVICE_HOST=0.0.0.0 \
    AVITO_HTTP_PROXY="http://prx-squid-rev.msk.avito.ru:9090" \
    AVITO_HTTPS_PROXY="http://prx-squid-rev.msk.avito.ru:9090" \
    SERVICE_PORT=8890

ENV PATH="$PROJECT_ROOT/bin:$PATH" PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

COPY . $PROJECT_ROOT
