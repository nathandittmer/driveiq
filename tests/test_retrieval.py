from __future__ import annotations

from driveiq.retrieval.embedder import get_embedding_provider
from driveiq.processing.chunker import build_chunks_for_document
from driveiq.retrieval.retrieve import retrieve_top_k
from driveiq.retrieval.vector_store import (
    build_stored_vector_record,
    write_vector_store,
)
from driveiq.schemas.document import DocumentMetadata, DocumentRecord


def make_document(document_id: str, text: str, filename: str) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        document_type="text",
        title=filename,
        text=text,
        metadata=DocumentMetadata(
            source_path=f"data/raw/{filename}",
            filename=filename,
            file_extension=".txt",
            extra={"parser": "plain_text", "normalization_applied": True},
        ),
    )


def test_retrieval_returns_ranked_results_from_local_vector_store(tmp_path) -> None:
    documents = [
        make_document("doc_search", "DriveIQ helps users search documents.", "search.txt"),
        make_document("doc_food", "This document is about cooking recipes.", "recipes.txt"),
    ]

    chunks = []
    for document in documents:
        chunks.extend(build_chunks_for_document(document, chunk_size=500, chunk_overlap=75))

    provider = get_embedding_provider()
    embeddings = provider.embed_texts([chunk.text for chunk in chunks])

    records = [
        build_stored_vector_record(chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=False)
    ]

    write_vector_store(records, str(tmp_path))

    response = retrieve_top_k(
        query="search documents",
        top_k=2,
        index_dir=str(tmp_path),
    )

    assert response.query == "search documents"
    assert response.top_k == 2
    assert len(response.results) == 2
    assert all(result.chunk_id for result in response.results)
    assert all(result.metadata.get("source_filename") for result in response.results)