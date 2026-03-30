from __future__ import annotations

from pathlib import Path

import fitz


def parse_pdf_file(path: Path) -> tuple[str, dict[str, str | int | list[dict[str, int | str]]]]:
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported PDF file type: {path}")

    doc = fitz.open(path)
    page_texts: list[str] = []
    page_metadata: list[dict[str, int | str]] = []

    try:
        for page_index, page in enumerate(doc):
            text = page.get_text("text").strip()
            page_texts.append(text)
            page_metadata.append(
                {
                    "page_number": page_index + 1,
                    "character_count": len(text),
                    "text_preview": text[:120],
                }
            )
    finally:
        doc.close()

    combined_text = "\n\n".join(text for text in page_texts if text).strip()

    metadata = {
        "parser": "pdf",
        "page_count": len(page_texts),
        "character_count": len(combined_text),
        "pages": page_metadata,
    }

    return combined_text, metadata