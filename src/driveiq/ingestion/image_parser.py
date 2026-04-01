from __future__ import annotations

from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def parse_image_file(path: Path) -> tuple[str, dict]:
    if not is_image_file(path):
        raise ValueError(f"Unsupported image file type: {path}")

    metadata = {
        "parser": "image_stub",
        "ocr_text_present": False,
        "caption_text_present": False,
        "character_count": 0,
        "modality": "image",
        "stub_note": "Image parser stub added. OCR or caption extraction can be integrated later.",
    }

    return "", metadata