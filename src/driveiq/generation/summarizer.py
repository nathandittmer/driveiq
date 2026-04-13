from __future__ import annotations

from uuid import uuid4

from driveiq.generation.prompts import build_summary_prompt
from driveiq.schemas.document import DocumentRecord
from driveiq.schemas.response import SummaryResponse


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
            "prompt_preview": prompt[:300],
        },
    )