from __future__ import annotations

from collections import Counter
from uuid import uuid4

from driveiq.generation.prompts import (
    CROSS_DOCUMENT_BRIEF_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    build_cross_document_brief_prompt,
    build_summary_prompt,
)
from driveiq.retrieval.retrieve import retrieve_top_k
from driveiq.schemas.document import DocumentRecord
from driveiq.schemas.response import RetrievedChunk, SummaryResponse


def simple_extractive_summary(text: str, max_sentences: int = 3) -> str:
    if not text.strip():
        return "No content available to summarize."

    normalized = text.replace("\n", " ").strip()
    sentence_candidates = [
        sentence.strip()
        for sentence in normalized.split(".")
        if sentence.strip()
    ]

    if not sentence_candidates:
        return normalized[:300].strip()

    selected = sentence_candidates[:max_sentences]
    summary = ". ".join(selected).strip()

    if summary and not summary.endswith("."):
        summary += "."

    return summary


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


def summarize_document(document: DocumentRecord) -> SummaryResponse:
    prompt = build_summary_prompt(
        document_text=document.text,
        document_title=document.title,
    )

    summary_text = simple_extractive_summary(document.text)

    return SummaryResponse(
        request_id=f"summary_{uuid4().hex[:12]}",
        document_id=document.document_id,
        summary_text=summary_text,
        supporting_chunk_ids=[],
        metadata={
            "generation_mode": "extractive_fallback",
            "document_title": document.title,
            "document_type": document.document_type,
            "source_filename": document.metadata.filename,
            "source_type": document.document_type,
            "character_count": len(document.text),
            "prompt_preview": prompt[:300],
            "prompt_version": SUMMARY_PROMPT_VERSION,
        },
    )


def simple_cross_document_brief(
    retrieved_chunks: list[RetrievedChunk],
    max_points: int = 3,
) -> str:
    if not retrieved_chunks:
        return "No relevant cross-document context was found."

    lines: list[str] = []
    seen_files: set[str] = set()

    for chunk in retrieved_chunks:
        source_filename = chunk.metadata.get("source_filename", "unknown")
        if source_filename in seen_files:
            continue

        seen_files.add(source_filename)
        preview = chunk.text.strip().replace("\n", " ")
        preview = preview[:160] + ("..." if len(preview) > 160 else "")
        lines.append(f"- {source_filename}: {preview}")

        if len(lines) >= max_points:
            break

    return "\n".join(lines) if lines else "No relevant cross-document context was found."


def build_cross_document_brief(
    user_goal: str,
    top_k: int = 5,
) -> SummaryResponse:
    retrieval_response = retrieve_top_k(user_goal, top_k=top_k)

    prompt = build_cross_document_brief_prompt(
        user_goal=user_goal,
        retrieved_chunks=retrieval_response.results,
    )

    brief_text = simple_cross_document_brief(retrieval_response.results)
    trace_metadata = build_trace_metadata(retrieval_response.results)

    return SummaryResponse(
        request_id=f"brief_{uuid4().hex[:12]}",
        document_id=None,
        summary_text=brief_text,
        supporting_chunk_ids=[chunk.chunk_id for chunk in retrieval_response.results],
        metadata={
            "generation_mode": "cross_document_fallback",
            "user_goal": user_goal,
            "prompt_preview": prompt[:300],
            "prompt_version": CROSS_DOCUMENT_BRIEF_PROMPT_VERSION,
            **trace_metadata,
        },
    )