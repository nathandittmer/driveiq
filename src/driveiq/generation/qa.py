from __future__ import annotations

from collections import Counter
from uuid import uuid4

from driveiq.generation.prompts import (
    GROUNDED_QA_PROMPT_VERSION,
    build_grounded_qa_prompt,
)
from driveiq.retrieval.retrieve import retrieve_top_k
from driveiq.schemas.response import QAResponse, RetrievedChunk


def simple_grounded_answer(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant information found in the available documents."

    top_chunk = chunks[0].text.strip()
    if not top_chunk:
        return "No relevant information found in the available documents."

    return top_chunk[:300] + ("..." if len(top_chunk) > 300 else "")


def build_trace_metadata(chunks: list[RetrievedChunk]) -> dict:
    source_filenames = [
        chunk.metadata.get("source_filename")
        for chunk in chunks
        if chunk.metadata.get("source_filename")
    ]
    source_types = [
        chunk.metadata.get("source_type")
        for chunk in chunks
        if chunk.metadata.get("source_type")
    ]

    filename_counts = Counter(source_filenames)
    source_type_counts = Counter(source_types)

    return {
        "supporting_chunk_count": len(chunks),
        "retrieved_sources": source_filenames,
        "retrieved_source_types": source_types,
        "top_source_files": filename_counts.most_common(3),
        "top_source_types": source_type_counts.most_common(3),
    }


def answer_question(query: str, top_k: int = 5) -> QAResponse:
    retrieval_response = retrieve_top_k(query, top_k=top_k)

    prompt = build_grounded_qa_prompt(
        query=query,
        retrieved_chunks=retrieval_response.results,
    )

    answer_text = simple_grounded_answer(retrieval_response.results)
    trace_metadata = build_trace_metadata(retrieval_response.results)

    return QAResponse(
        request_id=f"qa_{uuid4().hex[:12]}",
        query=query,
        answer_text=answer_text,
        supporting_chunk_ids=[chunk.chunk_id for chunk in retrieval_response.results],
        metadata={
            "generation_mode": "grounded_fallback",
            "prompt_version": GROUNDED_QA_PROMPT_VERSION,
            "prompt_preview": prompt[:300],
            "top_k": top_k,
            **trace_metadata,
        },
    )