from __future__ import annotations

from typing import Iterable

from driveiq.schemas.chunk import ChunkMetadata, ChunkRecord
from driveiq.schemas.document import DocumentRecord


def split_into_paragraphs(text: str) -> list[str]:
    if not text.strip():
        return []

    paragraphs = [part.strip() for part in text.split("\n\n")]
    return [paragraph for paragraph in paragraphs if paragraph]


def iter_paragraph_spans(text: str) -> Iterable[tuple[str, int, int]]:
    cursor = 0

    for paragraph in split_into_paragraphs(text):
        start = text.find(paragraph, cursor)
        if start == -1:
            continue

        end = start + len(paragraph)
        cursor = end

        yield paragraph, start, end


def build_chunks_for_document(document: DocumentRecord) -> list[ChunkRecord]:
    if not document.text.strip():
        return []

    parser_name = document.metadata.extra.get("parser")
    source_filename = document.metadata.filename

    chunks: list[ChunkRecord] = []

    for chunk_index, (chunk_text, start_char, end_char) in enumerate(
        iter_paragraph_spans(document.text)
    ):
        chunk = ChunkRecord(
            chunk_id=f"{document.document_id}_chunk_{chunk_index:03d}",
            document_id=document.document_id,
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                source_type=document.document_type,
                source_filename=source_filename,
                parser_name=parser_name,
                extra={},
            ),
        )
        chunks.append(chunk)

    return chunks


def build_chunks_for_documents(documents: list[DocumentRecord]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    for document in documents:
        chunks.extend(build_chunks_for_document(document))

    return chunks