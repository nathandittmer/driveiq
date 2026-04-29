from __future__ import annotations

import json
from pathlib import Path


def load_report(path: str) -> dict:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    return json.loads(report_path.read_text(encoding="utf-8"))


def extract_summary_scores(report: dict) -> dict[str, float]:
    summary = report.get("summary", {})

    return {
        "retrieval_hit_rate": float(summary.get("retrieval_hit_rate", 0.0)),
        "qa_pass_rate": float(summary.get("qa_pass_rate", 0.0)),
        "summary_pass_rate": float(summary.get("summary_pass_rate", 0.0)),
    }


def compare_reports(
    current_report_path: str,
    baseline_report_path: str,
) -> dict:
    current = load_report(current_report_path)
    baseline = load_report(baseline_report_path)

    current_scores = extract_summary_scores(current)
    baseline_scores = extract_summary_scores(baseline)

    deltas = {
        metric: current_scores[metric] - baseline_scores.get(metric, 0.0)
        for metric in current_scores
    }

    return {
        "comparison_type": "eval_regression",
        "baseline_report": baseline_report_path,
        "current_report": current_report_path,
        "baseline_scores": baseline_scores,
        "current_scores": current_scores,
        "deltas": deltas,
        "regression_detected": any(delta < 0 for delta in deltas.values()),
    }


def write_comparison_report(report: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")