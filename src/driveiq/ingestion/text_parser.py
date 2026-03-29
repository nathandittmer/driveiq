from __future__ import annotations

from pathlib import Path


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS


def normalize_text_content(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    return normalized


def normalize_markdown_content(text: str) -> str:
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()

        cleaned_lines.append(stripped if stripped else "")

    normalized = "\n".join(cleaned_lines)
    return normalize_text_content(normalized)


def parse_text_file(path: Path) -> tuple[str, dict[str, str | int]]:
    if not is_text_file(path):
        raise ValueError(f"Unsupported text file type: {path}")

    raw_text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".md":
        parsed_text = normalize_markdown_content(raw_text)
        parser_kind = "markdown"
    else:
        parsed_text = normalize_text_content(raw_text)
        parser_kind = "plain_text"

    metadata = {
        "parser": parser_kind,
        "character_count": len(parsed_text),
        "line_count": len(parsed_text.splitlines()) if parsed_text else 0,
    }

    return parsed_text, metadata