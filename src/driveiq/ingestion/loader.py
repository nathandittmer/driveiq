from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from driveiq.ingestion.image_parser import is_image_file, parse_image_file
from driveiq.ingestion.pdf_parser import parse_pdf_file
from driveiq.ingestion.text_parser import parse_text_file
from driveiq.ingestion.transcript_parser import looks_like_transcript, parse_transcript_file
from driveiq.processing.normalize import normalize_parsed_output
from driveiq.schemas.document import DocumentMetadata, DocumentRecord


SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


def infer_document_type(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        try:
            raw_text = path.read_text(encoding="utf-8")
            if looks_like_transcript(raw_text):
                return "transcript"
        except Exception:
            pass

    return SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unknown")


def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text_and_extra_metadata(path: Path) -> tuple[str, dict]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8")
        if looks_like_transcript(raw_text):
            parsed_text, parser_metadata = parse_transcript_file(path)
            return normalize_parsed_output(parsed_text, parser_metadata)

        parsed_text, parser_metadata = parse_text_file(path)
        return normalize_parsed_output(parsed_text, parser_metadata)

    if suffix == ".md":
        parsed_text, parser_metadata = parse_text_file(path)
        return normalize_parsed_output(parsed_text, parser_metadata)

    if suffix == ".pdf":
        parsed_text, parser_metadata = parse_pdf_file(path)
        return normalize_parsed_output(parsed_text, parser_metadata)

    if is_image_file(path):
        parsed_text, parser_metadata = parse_image_file(path)
        return normalize_parsed_output(parsed_text, parser_metadata)

    return "", {}


def build_document_record(path: Path) -> DocumentRecord:
    stat = path.stat()
    parsed_text, parser_metadata = extract_text_and_extra_metadata(path)

    metadata = DocumentMetadata(
        source_path=str(path.as_posix()),
        filename=path.name,
        file_extension=path.suffix.lower() or None,
        mime_type=None,
        file_size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        extra=parser_metadata,
    )

    return DocumentRecord(
        document_id=f"doc_{uuid4().hex[:12]}",
        document_type=infer_document_type(path),
        title=path.stem,
        text=parsed_text,
        metadata=metadata,
    )


def load_documents(raw_data_dir: str) -> list[DocumentRecord]:
    root = Path(raw_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    documents: list[DocumentRecord] = []

    for path in sorted(iter_supported_files(root)):
        documents.append(build_document_record(path))

    return documents


def iter_supported_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if is_supported_file(path):
            yield path