from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from driveiq.evaluation.dataset import load_intent_examples
from driveiq.retrieval.intent_classifier import (
    predict_intent,
    train_intent_classifier,
)


def run_intent_eval(dataset_path: str = "data/eval/intent_eval.json") -> dict:
    examples = load_intent_examples(dataset_path)
    model = train_intent_classifier(examples)

    results = []
    correct = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for example in examples:
        prediction = predict_intent(model, example.query)
        is_correct = prediction.predicted_intent == example.intent

        if is_correct:
            correct += 1

        confusion_counts[(example.intent, prediction.predicted_intent)] += 1

        results.append(
            {
                "example_id": example.example_id,
                "query": example.query,
                "expected_intent": example.intent,
                "predicted_intent": prediction.predicted_intent,
                "confidence": prediction.confidence,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(examples) if examples else 0.0

    confusion_matrix = [
        {
            "expected_intent": expected,
            "predicted_intent": predicted,
            "count": count,
        }
        for (expected, predicted), count in sorted(confusion_counts.items())
    ]

    return {
        "eval_type": "intent_classification",
        "example_count": len(examples),
        "accuracy": accuracy,
        "results": results,
        "confusion_matrix": confusion_matrix,
    }


def write_intent_eval_report(report: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    report = run_intent_eval()
    output_path = "artifacts/reports/intent_eval_report.json"
    write_intent_eval_report(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()