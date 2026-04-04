from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_index: int
    start_char: int
    end_char: int
    page_number: int | None = None
    section_name: str | None = None
    source_type: str | None = None
    source_filename: str | None = None
    parser_name: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: ChunkMetadata