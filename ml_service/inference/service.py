"""Inference layer abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import math
import numpy as np
import torch

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    ort = None  # type: ignore[assignment]

from transformers import PreTrainedTokenizerBase

from lib.tokenizer import build_tokenizer

__all__ = ["ONNXRuntimeService", "OnnxTextGenerator", "read_onnx_metadata", "PerplexityCalculator"]


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


def read_onnx_metadata(path: str) -> Dict[str, str]:
    """Read metadata properties stored in an ONNX model."""

    import onnx

    model = onnx.load(path)
    return {prop.key: prop.value for prop in model.metadata_props}


@dataclass
class OnnxTextGenerator:
    """Minimal text generator on top of an ONNXRuntime causal LM."""

    service: ONNXRuntimeService
    max_new_tokens: int = 50
    _past_init_keys: Optional[List[np.ndarray]] = None
    _past_init_values: Optional[List[np.ndarray]] = None
    _past_key_names: Optional[List[str]] = None
    _past_value_names: Optional[List[str]] = None

    @classmethod
    def from_checkpoint(
        cls,
        onnx_path: str,
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        providers: Optional[Iterable[str]] = None,
        max_new_tokens: int = 50,
    ) -> "OnnxTextGenerator":
        service = ONNXRuntimeService.from_checkpoint(
            onnx_path,
            tokenizer=tokenizer,
            providers=providers,
        )
        return cls(service=service, max_new_tokens=max_new_tokens)

    def _next_token(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> int:
        inputs = {
            self.service.input_name: input_ids,
            self.service.attention_mask_name: attention_mask,
        }
        outputs = self.service.session.run(None, inputs)
        logits = outputs[0]  # expected shape: (1, seq_len, vocab)
        next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        return next_id

    def predict(self, text: str) -> str:
        encoded = self.service.tokenizer(text, return_tensors="np")
        input_ids = encoded["input_ids"]  # shape: (1, seq_len)
        attn_mask = encoded.get("attention_mask", np.ones_like(input_ids))

        # Подготовка пустых past_key_values для первого шага
        if self._past_init_keys is None:
            self._past_init_keys = []
            self._past_init_values = []
            self._past_key_names = []
            self._past_value_names = []
            for inp in self.service.session.get_inputs():
                if inp.name.startswith("past_key_values") and ".key" in inp.name:
                    # inp.shape: [batch, num_heads, past_seq_len, head_dim]
                    _, num_heads, _, head_dim = inp.shape
                    self._past_init_keys.append(
                        np.zeros((1, int(num_heads), 0, int(head_dim)), dtype=np.float32)
                    )
                    self._past_key_names.append(inp.name)
                elif inp.name.startswith("past_key_values") and ".value" in inp.name:
                    _, num_heads, _, head_dim = inp.shape
                    self._past_init_values.append(
                        np.zeros((1, int(num_heads), 0, int(head_dim)), dtype=np.float32)
                    )
                    self._past_value_names.append(inp.name)

        past_keys = list(self._past_init_keys or [])
        past_values = list(self._past_init_values or [])

        generated: List[int] = []

        # Первый прогон: весь промпт, пустые past
        def _build_inputs(ids: np.ndarray, mask: np.ndarray, past_k, past_v, past_len: int):
            # position_ids = [past_len, past_len+1, ...]
            pos = np.arange(past_len, past_len + ids.shape[1], dtype=np.int64)[None, :]
            feed = {
                self.service.input_name: ids.astype(np.int64),
                self.service.attention_mask_name: mask.astype(np.int64),
                "position_ids": pos,
            }
            # past_key_values.X.{key,value}
            # inputs order matches session inputs to avoid missing keys
            for idx, key_name in enumerate(self._past_key_names or []):
                feed[key_name] = past_k[idx]
            for idx, val_name in enumerate(self._past_value_names or []):
                feed[val_name] = past_v[idx]
            return feed

        past_len = 0
        ids = input_ids
        mask = attn_mask

        for _ in range(self.max_new_tokens):
            feed = _build_inputs(ids, mask, past_keys, past_values, past_len)
            outputs = self.service.session.run(None, feed)

            logits = outputs[0]
            next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            generated.append(next_id)

            # Обновляем past_key_values из выходов present.*
            presents = outputs[1:]
            past_keys = presents[0::2]
            past_values = presents[1::2]

            # Готовим вход для следующего шага: только новый токен
            ids = np.array([[next_id]], dtype=np.int64)
            past_len = past_keys[0].shape[2]  # past_sequence_length (includes all prev tokens)
            mask = np.ones((1, past_len + 1), dtype=np.int64)

        full_ids = np.concatenate([input_ids, np.array([generated], dtype=np.int64)], axis=1)
        return self.service.tokenizer.decode(full_ids[0], skip_special_tokens=True)

    def predict_batch(self, texts: Iterable[str]) -> List[str]:
        return [self.predict(t) for t in texts]


@dataclass
class PerplexityCalculator:
    """Compute avg loss / perplexity using a HF causal LM model."""

    model: "torch.nn.Module"
    tokenizer: PreTrainedTokenizerBase
    device: "torch.device"
    max_length: int = 512
    batch_size: int = 4

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def compute(self, texts: Iterable[str]) -> Dict[str, float]:
        texts_list = list(texts)
        if not texts_list:
            return {"avg_loss": float("nan"), "perplexity": float("nan"), "count": 0}

        losses: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts_list), self.batch_size):
                batch = texts_list[i : i + self.batch_size]
                batch_tokens = self._tokenize(batch).to(self.device)
                # Используем labels = input_ids для оценки (classic LM)
                outputs = self.model(**batch_tokens, labels=batch_tokens["input_ids"])
                loss_val = outputs.loss
                losses.append(float(loss_val))

        avg_loss = sum(losses) / len(losses)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return {"avg_loss": avg_loss, "perplexity": ppl, "count": len(texts_list)}
