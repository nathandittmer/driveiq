from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedChunk] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    request_id: str
    document_id: str | None = None
    summary_text: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAResponse(BaseModel):
    request_id: str
    query: str
    answer_text: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)