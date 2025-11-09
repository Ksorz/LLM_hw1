"""Service for loading and running inference with trained models."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from lib.modeling import create_model
from lib.tokenizer import build_tokenizer

LOGGER = logging.getLogger(__name__)

__all__ = ["InferenceService", "load_model_for_inference"]


@dataclass
class InferenceService:
    """Service for running inference with a trained model."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.9

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> "InferenceService":
        """Load a trained model from checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        LOGGER.info("Загрузка модели из %s", model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        LOGGER.info("Модель загружена на %s", device)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device_obj,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @classmethod
    def from_scratch(
        cls,
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_config: Optional[Dict] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> "InferenceService":
        """Create a new untrained model (for testing/baseline)."""
        
        LOGGER.info("Создание новой необученной модели")
        
        tokenizer = tokenizer or build_tokenizer()
        model = create_model(tokenizer, model_config=model_config, bf16=False)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        LOGGER.info("Модель создана на %s", device)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device_obj,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def predict(self, text: str) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.predict(text) for text in texts]


def load_model_for_inference(
    checkpoint_path: Optional[str] = None,
    *,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 50,
) -> InferenceService:
    """
    Load a model for inference.
    
    If checkpoint_path is provided and exists, loads the trained model.
    Otherwise, creates a new untrained model.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        LOGGER.info("Загрузка обученной модели из %s", checkpoint_path)
        return InferenceService.from_pretrained(
            checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    else:
        if checkpoint_path:
            LOGGER.warning("Checkpoint %s не найден, создаём необученную модель", checkpoint_path)
        else:
            LOGGER.info("Создание необученной модели для baseline")
        
        return InferenceService.from_scratch(
            device=device,
            max_new_tokens=max_new_tokens,
        )

