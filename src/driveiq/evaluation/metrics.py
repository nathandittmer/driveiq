from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalEvalResult:
    example_id: str
    query: str
    expected_source_files: list[str]
    retrieved_source_files: list[str]
    hit: bool
    matched_source_files: list[str]


def compute_retrieval_hit(
    expected_source_files: list[str],
    retrieved_source_files: list[str],
) -> tuple[bool, list[str]]:
    expected = set(expected_source_files)
    retrieved = set(retrieved_source_files)

    matched = sorted(expected & retrieved)
    return bool(matched), matched


def compute_hit_rate(results: list[RetrievalEvalResult]) -> float:
    if not results:
        return 0.0

    hits = sum(1 for result in results if result.hit)
    return hits / len(results)