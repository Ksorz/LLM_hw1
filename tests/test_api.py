"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from ml_service import create_app
from ml_service.backend.dependencies import AppDependencies


@pytest.fixture
def test_app():
    """Create a test app with mock dependencies."""
    
    def mock_predict(text: str) -> str:
        return f"Generated: {text}"
    
    def mock_predict_batch(texts):
        return [mock_predict(t) for t in texts]
    
    def mock_metadata():
        return {"commit": "test", "date": "2025-11-09", "experiment": "test"}
    
    deps = AppDependencies(
        predict=mock_predict,
        predict_batch=mock_predict_batch,
        metadata=mock_metadata,
    )
    return create_app(deps)


@pytest.fixture
def client(test_app):
    """Create a test client."""
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

