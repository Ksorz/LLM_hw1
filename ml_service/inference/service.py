"""Inference layer abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    ort = None  # type: ignore[assignment]

from transformers import PreTrainedTokenizerBase

from lib.tokenizer import build_tokenizer

__all__ = ["ONNXRuntimeService"]


@dataclass
class ONNXRuntimeService:
    """Thin wrapper around ``onnxruntime.InferenceSession`` for text models."""

    session: "ort.InferenceSession"
    tokenizer: PreTrainedTokenizerBase
    input_name: str = "input_ids"
    attention_mask_name: str = "attention_mask"
    output_names: Optional[List[str]] = None

    @classmethod
    def from_checkpoint(
        cls,
        onnx_path: str,
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        providers: Optional[Iterable[str]] = None,
        session_options: Optional["ort.SessionOptions"] = None,
    ) -> "ONNXRuntimeService":
        """Instantiate the service from an ONNX checkpoint on disk."""

        if ort is None:  # pragma: no cover - optional dependency
            raise RuntimeError("onnxruntime is not installed")

        tokenizer = tokenizer or build_tokenizer()
        session = ort.InferenceSession(onnx_path, providers=list(providers or ["CPUExecutionProvider"]), options=session_options)
        return cls(session=session, tokenizer=tokenizer)

    def _encode(self, text: str) -> Mapping[str, np.ndarray]:
        encoded = self.tokenizer(text, return_tensors="np")
        return {
            self.input_name: encoded["input_ids"],
            self.attention_mask_name: encoded.get("attention_mask", np.ones_like(encoded["input_ids"])),
        }

    def _postprocess(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        if self.output_names:
            return dict(zip(self.output_names, outputs))
        return {f"output_{idx}": array for idx, array in enumerate(outputs)}

    def predict(self, text: str) -> Dict[str, np.ndarray]:
        """Run a single text through the ONNX session."""

        inputs = self._encode(text)
        outputs = self.session.run(self.output_names, inputs)
        return self._postprocess(outputs)

    def predict_batch(self, texts: Iterable[str]) -> List[Dict[str, np.ndarray]]:
        return [self.predict(text) for text in texts]
