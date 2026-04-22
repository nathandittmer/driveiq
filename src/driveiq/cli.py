from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from driveiq.config import get_settings
from driveiq.generation.qa import answer_question
from driveiq.generation.summarizer import summarize_document
from driveiq.ingestion.loader import load_documents
from driveiq.logging_utils import configure_logging, ensure_directory, get_logger
from driveiq.processing.chunker import build_chunks_for_documents
from driveiq.processing.metadata import write_processed_documents
from driveiq.retrieval.embedder import get_embedding_provider
from driveiq.retrieval.retrieve import retrieve_top_k
from driveiq.retrieval.vector_store import (
    build_stored_vector_record,
    write_vector_store,
)


def create_run_directory(base_dir: str, prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / "runs" / f"{prefix}_{timestamp}"
    return ensure_directory(run_dir)


def run_index_command() -> dict:
    settings = get_settings()
    documents = load_documents(settings.app.paths.raw_data_dir)
    write_processed_documents(
        documents=documents,
        output_dir=settings.app.paths.processed_data_dir,
    )

    chunks = build_chunks_for_documents(documents)
    provider = get_embedding_provider()
    embeddings = provider.embed_texts([chunk.text for chunk in chunks])

    records = [
        build_stored_vector_record(chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=False)
    ]

    vector_store_summary = write_vector_store(
        records=records,
        output_dir=settings.app.paths.index_data_dir,
    )

    return {
        "status": "indexed",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "vector_count": len(records),
        "embedding_provider": records[0].embedding_provider if records else None,
        "embedding_model_name": records[0].embedding_model_name if records else None,
        "index_output_dir": vector_store_summary["output_dir"],
        "index_file_path": vector_store_summary["file_path"],
    }


def run_search_command(query: str, top_k: int) -> dict:
    response = retrieve_top_k(query=query, top_k=top_k)
    return response.model_dump()


def run_summarize_command(filename: str | None, document_id: str | None) -> dict:
    documents = load_documents(get_settings().app.paths.raw_data_dir)

    matched_document = None

    if document_id:
        matched_document = next(
            (doc for doc in documents if doc.document_id == document_id),
            None,
        )

    if not matched_document and filename:
        matched_document = next(
            (doc for doc in documents if doc.metadata.filename == filename),
            None,
        )

    if not matched_document:
        raise ValueError("Document not found for the provided filename or document_id.")

    return summarize_document(matched_document).model_dump()


def run_ask_command(query: str, top_k: int) -> dict:
    response = answer_question(query=query, top_k=top_k)
    return response.model_dump()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driveiq",
        description="DriveIQ CLI for indexing, retrieval, summarization, and grounded Q&A.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Run the indexing pipeline")

    search_parser = subparsers.add_parser("search", help="Search indexed chunks")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, dest="top_k")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a document")
    summarize_parser.add_argument("--filename", type=str, default=None)
    summarize_parser.add_argument("--document-id", type=str, default=None, dest="document_id")

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question")
    ask_parser.add_argument("query", type=str, help="Question to answer")
    ask_parser.add_argument("--top-k", type=int, default=5, dest="top_k")

    return parser


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)
    logger = get_logger("driveiq.cli")

    parser = build_parser()
    args = parser.parse_args()

    logger.info("DriveIQ CLI command starting: %s", args.command)

    if args.command == "index":
        result = run_index_command()
    elif args.command == "search":
        result = run_search_command(query=args.query, top_k=args.top_k)
    elif args.command == "summarize":
        if not args.filename and not args.document_id:
            raise ValueError("Provide --filename or --document-id for summarize.")
        result = run_summarize_command(
            filename=args.filename,
            document_id=args.document_id,
        )
    elif args.command == "ask":
        result = run_ask_command(query=args.query, top_k=args.top_k)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2))
    logger.info("DriveIQ CLI command complete: %s", args.command)


if __name__ == "__main__":
    main()