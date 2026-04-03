from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from driveiq.logging_utils import ensure_directory
from driveiq.schemas.document import DocumentRecord


def write_processed_documents(
    documents: list[DocumentRecord],
    output_dir: str,
) -> dict[str, str | int]:
    output_path = ensure_directory(Path(output_dir))

    written_files: list[str] = []

    for document in documents:
        filename = f"{document.document_id}.json"
        file_path = output_path / filename
        file_path.write_text(
            json.dumps(document.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        written_files.append(filename)

    manifest = {
        "generate_at": datetime.now(tz=UTC).isoformat(),
        "document_count": len(documents),
        "files": written_files,
    }

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_path.as_posix()),
        "document_count": len(documents),
        "manifest_path": str(manifest_path.as_posix()),
    }