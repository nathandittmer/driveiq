from __future__ import annotations

from typing import Any

from driveiq.generation.qa import answer_question
from driveiq.generation.summarizer import build_cross_document_brief, summarize_document
from driveiq.ingestion.loader import load_documents
from driveiq.retrieval.intent_classifier import (
    load_and_train_default_intent_classifier,
    predict_intent,
)
from driveiq.retrieval.retrieve import retrieve_top_k


def _summarize_best_matching_document(query: str) -> dict[str, Any]:
    documents = load_documents("data/raw")

    for document in documents:
        if document.metadata.filename.lower() in query.lower():
            return summarize_document(document).model_dump()

    text_docs = [doc for doc in documents if doc.text.strip()]
    if not text_docs:
        return {"error": "No text-bearing documents available to summarize."}

    return summarize_document(text_docs[0]).model_dump()


def route_query(query: str, top_k: int = 5) -> dict[str, Any]:
    model = load_and_train_default_intent_classifier()
    prediction = predict_intent(model, query)
    intent = prediction.predicted_intent

    if intent == "search":
        result = retrieve_top_k(query=query, top_k=top_k).model_dump()
    elif intent == "ask":
        result = answer_question(query=query, top_k=top_k).model_dump()
    elif intent == "summarize":
        result = _summarize_best_matching_document(query).copy()
    elif intent == "brief":
        result = build_cross_document_brief(user_goal=query, top_k=top_k).model_dump()
    elif intent == "action_items":
        # Temporary fallback until action-item extraction gets its own module.
        result = build_cross_document_brief(user_goal=query, top_k=top_k).model_dump()
        result["metadata"]["temporary_route_note"] = (
            "action_items currently routes to cross-document brief fallback"
        )
    else:
        result = answer_question(query=query, top_k=top_k).model_dump()

    return {
        "query": query,
        "predicted_intent": intent,
        "intent_confidence": prediction.confidence,
        "result": result,
    }