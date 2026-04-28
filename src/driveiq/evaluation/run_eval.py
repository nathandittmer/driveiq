from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from driveiq.config import get_settings
from driveiq.evaluation.dataset import EvalExample, load_eval_examples
from driveiq.evaluation.judge import GenerationQualityResult, evaluate_generation_quality
from driveiq.evaluation.metrics import (
    RetrievalEvalResult,
    compute_hit_rate,
    compute_retrieval_hit,
)
from driveiq.generation.qa import answer_question
from driveiq.generation.summarizer import summarize_document
from driveiq.generation.prompts import(
    CROSS_DOCUMENT_BRIEF_PROMPT_VERSION,
    GROUNDED_QA_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
)
from driveiq.ingestion.loader import load_documents
from driveiq.retrieval.retrieve import retrieve_top_k


RETRIEVAL_EVAL_PATH = "data/eval/retrieval_eval.json"
QA_EVAL_PATH = "data/eval/qa_eval.json"
SUMMARY_EVAL_PATH = "data/eval/summary_eval.json"


def run_retrieval_eval(
    eval_path: str = RETRIEVAL_EVAL_PATH,
    top_k: int = 3,
) -> dict:
    examples = load_eval_examples(eval_path)
    results: list[RetrievalEvalResult] = []

    for example in examples:
        response = retrieve_top_k(example.query, top_k=top_k)
        retrieved_files = [
            result.metadata.get("source_filename")
            for result in response.results
            if result.metadata.get("source_filename")
        ]

        hit, matched_files = compute_retrieval_hit(
            expected_source_files=example.expected_source_files,
            retrieved_source_files=retrieved_files,
        )

        results.append(
            RetrievalEvalResult(
                example_id=example.example_id,
                query=example.query,
                expected_source_files=example.expected_source_files,
                retrieved_source_files=retrieved_files,
                hit=hit,
                matched_source_files=matched_files,
            )
        )

    return {
        "eval_type": "retrieval",
        "eval_path": eval_path,
        "top_k": top_k,
        "example_count": len(results),
        "hit_rate": compute_hit_rate(results),
        "results": [result.__dict__ for result in results],
    }


def run_qa_eval(
    eval_path: str = QA_EVAL_PATH,
    top_k: int = 5,
) -> dict:
    examples = load_eval_examples(eval_path)
    results: list[GenerationQualityResult] = []

    for example in examples:
        response = answer_question(example.query, top_k=top_k)
        retrieved_files = response.metadata.get("retrieved_sources", [])

        results.append(
            evaluate_generation_quality(
                example_id=example.example_id,
                task_type=example.task_type,
                query=example.query,
                output_text=response.answer_text,
                expected_source_files=example.expected_source_files,
                retrieved_source_files=retrieved_files,
            )
        )

    passed_count = sum(1 for result in results if result.passed)

    return {
        "eval_type": "qa",
        "eval_path": eval_path,
        "top_k": top_k,
        "example_count": len(results),
        "pass_rate": passed_count / len(results) if results else 0.0,
        "results": [result.__dict__ for result in results],
    }


def find_document_for_summary(example: EvalExample):
    documents = load_documents("data/raw")
    expected_files = set(example.expected_source_files)

    return next(
        (doc for doc in documents if doc.metadata.filename in expected_files),
        None,
    )


def run_summary_eval(
    eval_path: str = SUMMARY_EVAL_PATH,
) -> dict:
    examples = load_eval_examples(eval_path)
    results: list[GenerationQualityResult] = []

    for example in examples:
        document = find_document_for_summary(example)

        if not document:
            results.append(
                evaluate_generation_quality(
                    example_id=example.example_id,
                    task_type=example.task_type,
                    query=example.query,
                    output_text="",
                    expected_source_files=example.expected_source_files,
                    retrieved_source_files=[],
                )
            )
            continue

        response = summarize_document(document)

        results.append(
            evaluate_generation_quality(
                example_id=example.example_id,
                task_type=example.task_type,
                query=example.query,
                output_text=response.summary_text,
                expected_source_files=example.expected_source_files,
                retrieved_source_files=[document.metadata.filename],
            )
        )

    passed_count = sum(1 for result in results if result.passed)

    return {
        "eval_type": "summary",
        "eval_path": eval_path,
        "example_count": len(results),
        "pass_rate": passed_count / len(results) if results else 0.0,
        "results": [result.__dict__ for result in results],
    }


def build_combined_eval_report() -> dict:
    settings = get_settings()
    top_k = settings.retrieval.retrieval.top_k

    retrieval_report = run_retrieval_eval(top_k=top_k)
    qa_report = run_qa_eval(top_k=top_k)
    summary_report = run_summary_eval()

    return {
        "eval_type": "combined",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": {
            "top_k": top_k,
            "embedding_provider": settings.retrieval.embedding.provider,
            "embedding_model_name": settings.retrieval.embedding.embedding_model_name,
            "chunk_size": settings.retrieval.chunking.chunk_size,
            "chunk_overlap": settings.retrieval.chunking.chunk_overlap,
            "prompt_versions": {
                "summary": SUMMARY_PROMPT_VERSION,
                "grounded_qa": GROUNDED_QA_PROMPT_VERSION,
                "cross_document_brief": CROSS_DOCUMENT_BRIEF_PROMPT_VERSION,
            },
        },
        "summary": {
            "retrieval_hit_rate": retrieval_report["hit_rate"],
            "qa_pass_rate": qa_report["pass_rate"],
            "summary_pass_rate": summary_report["pass_rate"],
        },
        "retrieval": retrieval_report,
        "qa": qa_report,
        "summary_eval": summary_report,
    }


def write_eval_report(report: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    report = build_combined_eval_report()
    output_path = "artifacts/reports/combined_eval_report.json"
    write_eval_report(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()