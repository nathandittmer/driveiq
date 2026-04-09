from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from driveiq.logging_utils import ensure_directory
from driveiq.retrieval.embedder import EmbeddingResult
from driveiq.schemas.chunk import ChunkRecord


@dataclass
class StoredVectorRecord:
    chunk_id: str
    document_id: str
    text: str
    vector: list[float]
    embedding_provider: str
    embedding_model_name: str
    source_filename: str | None
    source_type: str | None
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


def build_stored_vector_record(
    chunk: ChunkRecord,
    embedding: EmbeddingResult,
) -> StoredVectorRecord:
    return StoredVectorRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        vector=embedding.vector,
        embedding_provider=embedding.provider,
        embedding_model_name=embedding.model_name,
        source_filename=chunk.metadata.source_filename,
        source_type=chunk.metadata.source_type,
        chunk_index=chunk.metadata.chunk_index,
        start_char=chunk.metadata.start_char,
        end_char=chunk.metadata.end_char,
        metadata={
            "section_name": chunk.metadata.section_name,
            "parser_name": chunk.metadata.parser_name,
            "extra": chunk.metadata.extra,
        },
    )


def write_vector_store(
    records: list[StoredVectorRecord],
    output_dir: str,
    filename: str = "vectors.json",
) -> dict[str, str | int]:
    output_path = ensure_directory(Path(output_dir))
    file_path = output_path / filename

    payload = {
        "record_count": len(records),
        "records": [asdict(record) for record in records],
    }

    file_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_path.as_posix()),
        "file_path": str(file_path.as_posix()),
        "record_count": len(records),
    }


def load_vector_store(
    input_dir: str,
    filename: str = "vectors.json",
) -> list[StoredVectorRecord]:
    file_path = Path(input_dir) / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Vector store file not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])

    return [StoredVectorRecord(**record) for record in records]