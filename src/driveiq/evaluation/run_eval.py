from __future__ import annotations

import json
from pathlib import Path

from driveiq.config import get_settings
from driveiq.evaluation.dataset import load_eval_examples
from driveiq.evaluation.metrics import (
    RetrievalEvalResult,
    compute_hit_rate,
    compute_retrieval_hit,
)
from driveiq.retrieval.retrieve import retrieve_top_k


def run_retrieval_eval(
    eval_path: str = "data/eval/retrieval_eval.json",
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
        "top_k": top_k,
        "example_count": len(results),
        "hit_rate": compute_hit_rate(results),
        "results": [result.__dict__ for result in results],
    }


def write_eval_report(report: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    report = run_retrieval_eval(top_k=settings.retrieval.retrieval.top_k)
    output_path = "artifacts/reports/retrieval_eval_report.json"
    write_eval_report(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()