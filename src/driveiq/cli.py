from __future__ import annotations

from datetime import datetime
from pathlib import Path

from driveiq.config import get_settings
from driveiq.ingestion.loader import load_documents
from driveiq.logging_utils import configure_logging, ensure_directory, get_logger
from driveiq.processing.metadata import write_processed_documents


def create_run_directory(base_dir: str, prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / "runs" / f"{prefix}_{timestamp}"
    return ensure_directory(run_dir)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)
    logger = get_logger("driveiq.cli")

    artifacts_dir = settings.app.paths.artifacts_dir
    run_dir = create_run_directory(artifacts_dir, prefix="smoke")

    logger.info("DriveIQ CLI smoke test starting")
    logger.info("Environment: %s", settings.app.environment)
    logger.info("Artifacts dir: %s", artifacts_dir)
    logger.info("Created run directory: %s", run_dir)

    documents = load_documents(settings.app.paths.raw_data_dir)
    logger.info("Loaded %d document(s) from raw data directory", len(documents))

    processed_summary = write_processed_documents(
        documents=documents,
        output_dir=settings.app.paths.processed_data_dir,
    )
    logger.info(
        "Wrote %d processed document artifact(s) to %s",
        processed_summary["document_count"],
        processed_summary["output_dir"],
    )
    logger.info("Processed manifest: %s", processed_summary["manifest_path"])

    marker_file = run_dir / "run_info.txt"
    marker_lines = [
        f"app_name={settings.app.app_name}",
        f"environment={settings.app.environment}",
        f"run_dir={run_dir}",
        f"document_count={len(documents)}",
        f"processed_output_dir={processed_summary['output_dir']}",
        f"processed_manifest={processed_summary['manifest_path']}",
    ]

    for doc in documents:
        logger.info(
            "Document loaded | id=%s | type=%s | file=%s",
            doc.document_id,
            doc.document_type,
            doc.metadata.filename,
        )
        marker_lines.append(
            f"document={doc.document_id}|{doc.document_type}|{doc.metadata.filename}"
        )

    marker_file.write_text("\n".join(marker_lines), encoding="utf-8")

    logger.info("Wrote marker file: %s", marker_file)
    logger.info("DriveIQ CLI smoke test complete")


if __name__ == "__main__":
    main()