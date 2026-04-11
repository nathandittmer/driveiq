from __future__ import annotations

import re

from driveiq.retrieval.vector_store import StoredVectorRecord


TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def compute_rerank_boost(query: str, record: StoredVectorRecord) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    boost = 0.0

    filename_tokens = tokenize(record.source_filename or "")
    section_tokens = tokenize(record.metadata.get("section_name") or "")
    chunk_tokens = tokenize(record.text[:500])

    filename_overlap = len(query_tokens & filename_tokens)
    section_overlap = len(query_tokens & section_tokens)
    chunk_overlap = len(query_tokens & chunk_tokens)

    boost += 0.03 * filename_overlap
    boost += 0.02 * section_overlap
    boost += 0.01 * chunk_overlap

    transcript_speaker_hint = (
        record.metadata.get("extra", {}).get("transcript_speaker_hint")
    )
    if transcript_speaker_hint:
        speaker_tokens = tokenize(transcript_speaker_hint)
        if query_tokens & speaker_tokens:
            boost += 0.05

    return boost


def rerank_scored_records(
    query: str,
    scored_records: list[tuple[StoredVectorRecord, float]],
) -> list[tuple[StoredVectorRecord, float]]:
    reranked: list[tuple[StoredVectorRecord, float]] = []

    for record, base_score in scored_records:
        boost = compute_rerank_boost(query, record)
        reranked.append((record, base_score + boost))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked