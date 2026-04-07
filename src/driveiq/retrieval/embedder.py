from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from driveiq.config import get_settings


@dataclass
class EmbeddingResult:
    text: str
    vector: list[float]
    model_name: str
    provider: str


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> EmbeddingResult:
        ...

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        ...


class DummyEmbeddingProvider:
    def __init__(self, model_name: str, provider_name: str = "dummy") -> None:
        self.model_name = model_name
        self.provider_name = provider_name

    def embed_text(self, text: str) -> EmbeddingResult:
        # Placeholder deterministic vector for interface testing only.
        length = float(len(text))
        word_count = float(len(text.split()))
        line_count = float(len(text.splitlines()))
        char_mod = float(len(text) % 10)

        return EmbeddingResult(
            text=text,
            vector=[length, word_count, line_count, char_mod],
            model_name=self.model_name,
            provider=self.provider_name,
        )

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    provider_name = settings.retrieval.embedding.provider
    model_name = settings.retrieval.embedding.embedding_model_name

    # For Day 16, return a dummy provider even if config says sentence_transformers.
    # This keeps the interface stable while the real model is added on Day 17.
    return DummyEmbeddingProvider(
        model_name=model_name,
        provider_name=f"{provider_name}_stub",
    )