from __future__ import annotations

import math

from driveiq.config import get_settings
from driveiq.retrieval.embedder import get_embedding_provider
from driveiq.retrieval.vector_store import StoredVectorRecord, load_vector_store
from driveiq.schemas.response import RetrievedChunk, SearchResponse


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def score_query_against_records(
    query_vector: list[float],
    records: list[StoredVectorRecord],
) -> list[tuple[StoredVectorRecord, float]]:
    scored_records: list[tuple[StoredVectorRecord, float]] = []

    for record in records:
        score = cosine_similarity(query_vector, record.vector)
        scored_records.append((record, score))

    scored_records.sort(key=lambda item: item[1], reverse=True)
    return scored_records


def retrieve_top_k(
    query: str,
    top_k: int | None = None,
    index_dir: str = "data/index",
) -> SearchResponse:
    settings = get_settings()
    effective_top_k = top_k or settings.retrieval.retrieval.top_k

    provider = get_embedding_provider()
    query_embedding = provider.embed_text(query)
    records = load_vector_store(index_dir)

    scored_records = score_query_against_records(query_embedding.vector, records)
    top_records = scored_records[:effective_top_k]

    results = [
        RetrievedChunk(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            text=record.text,
            score=score,
            metadata={
                "source_filename": record.source_filename,
                "source_type": record.source_type,
                "chunk_index": record.chunk_index,
                "start_char": record.start_char,
                "end_char": record.end_char,
                "embedding_provider": record.embedding_provider,
                "embedding_model_name": record.embedding_model_name,
                "section_name": record.metadata.get("section_name"),
                "parser_name": record.metadata.get("parser_name"),
                "extra": record.metadata.get("extra", {}),
            },
        )
        for record, score in top_records
    ]

    return SearchResponse(
        query=query,
        top_k=effective_top_k,
        results=results,
    )