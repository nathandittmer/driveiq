from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationQualityResult:
    example_id: str
    task_type: str
    query: str
    output_text: str
    expected_source_files: list[str]
    retrieved_source_files: list[str]
    has_output: bool
    source_hit: bool
    passed: bool


def has_non_empty_output(output_text: str) -> bool:
    return bool(output_text and output_text.strip())


def expected_source_hit(
    expected_source_files: list[str],
    retrieved_source_files: list[str],
) -> bool:
    return bool(set(expected_source_files) & set(retrieved_source_files))


def evaluate_generation_quality(
    example_id: str,
    task_type: str,
    query: str,
    output_text: str,
    expected_source_files: list[str],
    retrieved_source_files: list[str],
) -> GenerationQualityResult:
    has_output = has_non_empty_output(output_text)
    source_hit = expected_source_hit(expected_source_files, retrieved_source_files)

    return GenerationQualityResult(
        example_id=example_id,
        task_type=task_type,
        query=query,
        output_text=output_text,
        expected_source_files=expected_source_files,
        retrieved_source_files=retrieved_source_files,
        has_output=has_output,
        source_hit=source_hit,
        passed=has_output and source_hit,
    )