from __future__ import annotations

from dataclasses import dataclass
import re

from driveiq.config import get_settings
from driveiq.schemas.chunk import ChunkMetadata, ChunkRecord
from driveiq.schemas.document import DocumentRecord


@dataclass
class TextSpan:
    text: str
    start_char: int
    end_char: int


SECTION_LINE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9\s&/\-]{1,80}$")
TRANSCRIPT_SPEAKER_PATTERN = re.compile(r"^([A-Za-z0-9 _.-]+)\s+\(\d{2}:\d{2}\):")


def split_into_paragraph_spans(text: str) -> list[TextSpan]:
    if not text.strip():
        return []

    spans: list[TextSpan] = []
    cursor = 0

    for paragraph in [part.strip() for part in text.split("\n\n") if part.strip()]:
        start = text.find(paragraph, cursor)
        if start == -1:
            continue

        end = start + len(paragraph)
        spans.append(TextSpan(text=paragraph, start_char=start, end_char=end))
        cursor = end

    return spans


def build_windowed_spans(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextSpan]:
    if not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    spans: list[TextSpan] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            spans.append(
                TextSpan(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return spans


def merge_paragraphs_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextSpan]:
    paragraph_spans = split_into_paragraph_spans(text)

    if not paragraph_spans:
        return []

    merged_chunks: list[TextSpan] = []
    current_parts: list[TextSpan] = []

    for paragraph in paragraph_spans:
        if len(paragraph.text) > chunk_size:
            if current_parts:
                merged_chunks.append(_combine_spans(current_parts))
                current_parts = []

            merged_chunks.extend(
                build_windowed_spans(
                    paragraph.text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
            continue

        candidate_parts = current_parts + [paragraph]
        candidate_text = "\n\n".join(part.text for part in candidate_parts)

        if len(candidate_text) <= chunk_size:
            current_parts = candidate_parts
        else:
            if current_parts:
                merged_chunks.append(_combine_spans(current_parts))

            overlap_parts = _select_overlap_parts(current_parts, chunk_overlap)
            current_parts = overlap_parts + [paragraph]

            candidate_text = "\n\n".join(part.text for part in current_parts)
            if len(candidate_text) > chunk_size:
                merged_chunks.append(_combine_spans([paragraph]))
                current_parts = []

    if current_parts:
        merged_chunks.append(_combine_spans(current_parts))

    return merged_chunks


def _combine_spans(parts: list[TextSpan]) -> TextSpan:
    combined_text = "\n\n".join(part.text for part in parts)
    start_char = parts[0].start_char
    end_char = parts[-1].end_char

    return TextSpan(
        text=combined_text,
        start_char=start_char,
        end_char=end_char,
    )


def _select_overlap_parts(parts: list[TextSpan], chunk_overlap: int) -> list[TextSpan]:
    if not parts or chunk_overlap <= 0:
        return []

    selected: list[TextSpan] = []
    accumulated_length = 0

    for part in reversed(parts):
        selected.insert(0, part)
        accumulated_length += len(part.text)
        if accumulated_length >= chunk_overlap:
            break

    return selected


def infer_section_name(chunk_text: str) -> str | None:
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0]
    if SECTION_LINE_PATTERN.match(first_line) and len(first_line.split()) <= 8:
        return first_line

    return None


def infer_transcript_speaker(chunk_text: str) -> str | None:
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not lines:
        return None

    match = TRANSCRIPT_SPEAKER_PATTERN.match(lines[0])
    if match:
        return match.group(1).strip()

    return None


def build_chunk_extra_metadata(
    document: DocumentRecord,
    chunk_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    parser_name = document.metadata.extra.get("parser")
    section_name = infer_section_name(chunk_text)
    transcript_speaker = (
        infer_transcript_speaker(chunk_text)
        if document.document_type == "transcript"
        else None
    )

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_title": document.title,
        "file_extension": document.metadata.file_extension,
        "normalization_applied": document.metadata.extra.get("normalization_applied"),
        "section_name_hint": section_name,
        "transcript_speaker_hint": transcript_speaker,
    }


def build_chunks_for_document(
    document: DocumentRecord,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkRecord]:
    if not document.text.strip():
        return []

    settings = get_settings()
    chunk_size = chunk_size or settings.retrieval.chunking.chunk_size
    chunk_overlap = chunk_overlap or settings.retrieval.chunking.chunk_overlap

    parser_name = document.metadata.extra.get("parser")
    source_filename = document.metadata.filename

    text_spans = merge_paragraphs_into_chunks(
        document.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[ChunkRecord] = []

    for chunk_index, span in enumerate(text_spans):
        section_name = infer_section_name(span.text)

        chunk = ChunkRecord(
            chunk_id=f"{document.document_id}_chunk_{chunk_index:03d}",
            document_id=document.document_id,
            text=span.text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=span.start_char,
                end_char=span.end_char,
                page_number=None,
                section_name=section_name,
                source_type=document.document_type,
                source_filename=source_filename,
                parser_name=parser_name,
                extra=build_chunk_extra_metadata(
                    document=document,
                    chunk_text=span.text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
            ),
        )
        chunks.append(chunk)

    return chunks


def build_chunks_for_documents(
    documents: list[DocumentRecord],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    for document in documents:
        chunks.extend(
            build_chunks_for_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks