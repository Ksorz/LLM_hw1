"""Runtime dependencies for the FastAPI application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Callable, Optional


class SinglePredictFn(Protocol):
    def __call__(self, text: str) -> str:  # pragma: no cover - protocol
        ...


class BatchPredictFn(Protocol):
    def __call__(self, texts: Iterable[str]) -> List[str]:  # pragma: no cover - protocol
        ...


class MetadataFn(Protocol):
    def __call__(self) -> Dict[str, str]:  # pragma: no cover - protocol
        ...


EvaluateFn = Callable[[Iterable[str]], Dict[str, float]]


@dataclass
class AppDependencies:
    predict: SinglePredictFn
    predict_batch: BatchPredictFn
    metadata: MetadataFn
    evaluate_perplexity: Optional[EvaluateFn] = None
