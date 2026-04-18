from __future__ import annotations

from fastapi import FastAPI

from driveiq.config import get_settings
from driveiq.ingestion.loader import load_documents
from driveiq.processing.chunker import build_chunks_for_documents
from driveiq.retrieval.embedder import get_embedding_provider
from driveiq.retrieval.vector_store import (
    build_stored_vector_record,
    write_vector_store,
)


settings = get_settings()

app = FastAPI(
    title=settings.app.app_name,
    version="0.1.0",
    description="DriveIQ API for multimodal retrieval and summarization workflows.",
)


@app.get("/")
def read_root() -> dict:
    return {
        "app_name": settings.app.app_name,
        "environment": settings.app.environment,
        "status": "ok",
        "message": "DriveIQ API is running.",
    }


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "app_name": settings.app.app_name,
        "environment": settings.app.environment,
    }


def run_indexing_pipeline() -> dict:
    documents = load_documents(settings.app.paths.raw_data_dir)
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


@app.post("/index")
def index_documents() -> dict:
    return run_indexing_pipeline()