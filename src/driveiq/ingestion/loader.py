from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable
from uuid import uuid4

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
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unknown")

def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS

def read_text_if_supported(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    return ""

def build_document_record(path: Path) -> DocumentRecord:
    stat = path.stat()

    metadata = DocumentMetadata(
        source_path=str(path.as_posix()),
        filename=path.name,
        file_extension=path.suffix.lower() or None,
        mime_type=None,
        file_size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        extra={},
    )

    return DocumentRecord(
        document_id=f"doc_{uuid4().hex[:12]}",
        document_type=infer_document_type(path),
        title=path.stem,
        text=read_text_if_supported(path),
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