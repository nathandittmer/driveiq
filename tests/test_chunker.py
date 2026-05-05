from __future__ import annotations

from driveiq.processing.chunker import build_chunks_for_document
from driveiq.schemas.document import DocumentMetadata, DocumentRecord


def make_document(text: str) -> DocumentRecord:
    return DocumentRecord(
        document_id="doc_test",
        document_type="text",
        title="Test Document",
        text=text,
        metadata=DocumentMetadata(
            source_path="data/raw/test.txt",
            filename="test.txt",
            file_extension=".txt",
            extra={
                "parser": "plain_text",
                "normalization_applied": True,
            },
        ),
    )


def test_empty_document_produces_no_chunks() -> None:
    document = make_document("")
    chunks = build_chunks_for_document(document)

    assert chunks == []


def test_short_document_produces_single_chunk() -> None:
    document = make_document("This is a short document about retrieval.")
    chunks = build_chunks_for_document(document, chunk_size=500, chunk_overlap=75)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_test"
    assert chunks[0].metadata.source_filename == "test.txt"
    assert chunks[0].metadata.parser_name == "plain_text"


def test_paragraphs_are_grouped_into_chunks() -> None:
    text = "First paragraph about search.\n\nSecond paragraph about summaries."
    document = make_document(text)

    chunks = build_chunks_for_document(document, chunk_size=500, chunk_overlap=75)

    assert len(chunks) == 1
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[0].text


def test_small_chunk_size_creates_multiple_chunks() -> None:
    text = (
        "First paragraph about search quality.\n\n"
        "Second paragraph about summarization quality.\n\n"
        "Third paragraph about evaluation quality."
    )
    document = make_document(text)

    chunks = build_chunks_for_document(document, chunk_size=70, chunk_overlap=20)

    assert len(chunks) >= 2
    assert all(chunk.metadata.extra["chunk_size"] == 70 for chunk in chunks)
    assert all(chunk.metadata.extra["chunk_overlap"] == 20 for chunk in chunks)


def test_chunk_metadata_preserves_source_context() -> None:
    document = make_document("Customer FAQ\n\nDriveIQ helps users search documents.")

    chunks = build_chunks_for_document(document, chunk_size=500, chunk_overlap=75)

    assert chunks[0].metadata.source_type == "text"
    assert chunks[0].metadata.source_filename == "test.txt"
    assert chunks[0].metadata.extra["document_title"] == "Test Document"
    assert chunks[0].metadata.extra["file_extension"] == ".txt"
    assert chunks[0].metadata.extra["normalization_applied"] is True