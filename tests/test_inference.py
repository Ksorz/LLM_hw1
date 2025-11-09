"""Tests for the inference service."""

import pytest
import torch

from lib.inference import InferenceService


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inference_service_from_scratch_cuda():
    """Test creating inference service from scratch on CUDA."""
    service = InferenceService.from_scratch(device="cuda", max_new_tokens=10)
    
    assert service.device.type == "cuda"
    assert service.model is not None
    assert service.tokenizer is not None
    
    # Test prediction
    result = service.predict("Test")
    assert isinstance(result, str)
    assert len(result) > 0


def test_inference_service_from_scratch_cpu():
    """Test creating inference service from scratch on CPU."""
    service = InferenceService.from_scratch(device="cpu", max_new_tokens=10)
    
    assert service.device.type == "cpu"
    assert service.model is not None
    assert service.tokenizer is not None
    
    # Test prediction
    result = service.predict("Test")
    assert isinstance(result, str)
    assert len(result) > 0


def test_inference_service_batch():
    """Test batch prediction."""
    service = InferenceService.from_scratch(device="cpu", max_new_tokens=5)
    
    texts = ["A", "B", "C"]
    results = service.predict_batch(texts)
    
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)

