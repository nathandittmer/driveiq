from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentType = Literal["pdf", "text", "image", "transcript", "unknown"]


class DocumentMetadata(BaseModel):
    source_path: str
    filename: str
    file_extension: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    document_id: str
    document_type: DocumentType = "unknown"
    title: str | None = None
    text: str = ""
    metadata: DocumentMetadata
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)