from __future__ import annotations

from fastapi.testclient import TestClient

from driveiq.api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["app_name"] == "DriveIQ"


def test_index_endpoint() -> None:
    response = client.post("/index")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "indexed"
    assert payload["document_count"] >= 1
    assert payload["chunk_count"] >= 1
    assert payload["vector_count"] >= 1


def test_search_endpoint() -> None:
    client.post("/index")

    response = client.post(
        "/search",
        json={
            "query": "What content types does DriveIQ support?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What content types does DriveIQ support?"
    assert payload["top_k"] == 3
    assert len(payload["results"]) <= 3


def test_summarize_endpoint() -> None:
    response = client.post(
        "/summarize",
        json={"filename": "product_brief.pdf"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary_text"]
    assert payload["metadata"]["source_filename"] == "product_brief.pdf"


def test_ask_endpoint() -> None:
    client.post("/index")

    response = client.post(
        "/ask",
        json={
            "query": "What are the next steps for DriveIQ?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_text"]
    assert payload["query"] == "What are the next steps for DriveIQ?"
    assert payload["metadata"]["supporting_chunk_count"] <= 3