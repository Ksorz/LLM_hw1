"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from ml_service import create_app
from ml_service.backend.dependencies import AppDependencies
from ml_service.backend.database import get_db
from ml_service.backend.models import DatasetSample, Experiment
import io


@pytest.fixture
def test_app():
    """Create a test app with mock dependencies."""
    
    def mock_predict(text: str) -> str:
        return f"Generated: {text}"
    
    def mock_predict_batch(texts):
        return [mock_predict(t) for t in texts]
    
    def mock_metadata():
        return {"commit": "test", "date": "2025-11-09", "experiment": "test"}

    def mock_evaluate(texts):
        return {"avg_loss": 1.0, "perplexity": 2.0, "count": len(texts)}
    
    deps = AppDependencies(
        predict=mock_predict,
        predict_batch=mock_predict_batch,
        metadata=mock_metadata,
        evaluate_perplexity=mock_evaluate,
    )
    return create_app(deps)


@pytest.fixture
def client(test_app):
    """Create a test client."""
    def _dummy_db():
        # Избегаем реального подключения к Postgres в тестах
        yield None

    test_app.dependency_overrides[get_db] = _dummy_db
    return TestClient(test_app)


def test_forward_valid(client):
    """Test /forward with valid input."""
    response = client.post("/forward", json={"text": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "Generated: Hello"


def test_forward_empty_text(client):
    """Test /forward with empty text."""
    response = client.post("/forward", json={"text": ""})
    assert response.status_code == 422


def test_forward_missing_field(client):
    """Test /forward with missing field."""
    response = client.post("/forward", json={"wrong": "field"})
    assert response.status_code == 422


def test_forward_batch_valid(client):
    """Test /forward_batch with valid input."""
    response = client.post("/forward_batch", json={"texts": ["A", "B"]})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_forward_batch_empty(client):
    """Test /forward_batch with empty list."""
    response = client.post("/forward_batch", json={"texts": []})
    assert response.status_code == 422


def test_metadata(client):
    """Test /metadata endpoint."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "commit" in data
    assert "date" in data
    assert "experiment" in data


def test_health(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class _FakeQuery:
    def __init__(self, result=None):
        self._result = result

    def filter(self, *_, **__):
        return self

    def first(self):
        return self._result


class _FakeSession:
    def __init__(self, result=None):
        self.result = result
        self.saved = []
        self.added = []

    def bulk_save_objects(self, objs):
        self.saved.extend(objs)

    def add(self, obj):
        self.added.append(obj)
        if isinstance(obj, Experiment) and obj.id is None:
            obj.id = 1

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def query(self, model):
        return _FakeQuery(self.result)

    def close(self):
        pass


@pytest.fixture
def client_with_db(test_app):
    fake_db = _FakeSession()

    def _fake_db():
        yield fake_db

    test_app.dependency_overrides[get_db] = _fake_db
    return TestClient(test_app), fake_db


def test_add_data_csv(client_with_db):
    client, fake_db = client_with_db
    csv_content = "text,label\nhello,test\n"
    files = {"file": ("data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    resp = client.put("/add_data", files=files)
    assert resp.status_code == 200
    assert resp.json()["inserted"] == 1
    assert isinstance(fake_db.saved[0], DatasetSample)


def test_metrics_not_found(client_with_db):
    client, fake_db = client_with_db
    fake_db.result = None
    resp = client.get("/metrics/123")
    assert resp.status_code == 404


def test_evaluate_json(client):
    response = client.post("/evaluate", json={"texts": ["hello", "world"]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["avg_loss"] == 1.0
    assert data["perplexity"] == 2.0
    assert "duration_ms" in data


def test_evaluate_empty(client):
    response = client.post("/evaluate", json={"texts": []})
    assert response.status_code == 422

