from __future__ import annotations

import re
from typing import Any


MULTISPACE_PATTERN = re.compile(r"[ \t]+")
MULTIBLANKLINE_PATTERN = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized_lines: list[str] = []

    for line in text.splitlines():
        cleaned = MULTISPACE_PATTERN.sub(" ", line).strip()
        normalized_lines.append(cleaned)

    normalized = "\n".join(normalized_lines).strip()
    normalized = MULTIBLANKLINE_PATTERN.sub("\n\n", normalized)

    return normalized


def build_normalization_metadata(
    original_text: str,
    normalized_text: str,
) -> dict[str, Any]:
    return {
        "normalization_applied": True,
        "original_character_count": len(original_text),
        "normalized_character_count": len(normalized_text),
        "text_was_changed": original_text != normalized_text,
    }


def normalize_parsed_output(
    text: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    base_metadata = dict(metadata or {})
    normalized_text = normalize_text(text)
    normalization_metadata = build_normalization_metadata(text, normalized_text)
    base_metadata.update(normalization_metadata)
    return normalized_text, base_metadata