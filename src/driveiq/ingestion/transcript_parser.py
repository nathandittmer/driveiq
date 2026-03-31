from __future__ import annotations

import re
from pathlib import Path


TRANSCRIPT_LINE_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{2}:\d{2})\]\s+(?P<speaker>[^:]+):\s+(?P<utterance>.+)$"
)


def looks_like_transcript(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    matched = sum(1 for line in lines if TRANSCRIPT_LINE_PATTERN.match(line))
    return matched >= max(1, len(lines) // 2)


def parse_transcript_file(path: Path) -> tuple[str, dict]:
    raw_text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    segments: list[dict[str, str]] = []
    normalized_lines: list[str] = []

    for line in lines:
        match = TRANSCRIPT_LINE_PATTERN.match(line)
        if match:
            timestamp = match.group("timestamp")
            speaker = match.group("speaker").strip()
            utterance = match.group("utterance").strip()

            segments.append(
                {
                    "timestamp": timestamp,
                    "speaker": speaker,
                    "utterance": utterance,
                }
            )
            normalized_lines.append(f"{speaker} ({timestamp}): {utterance}")
        else:
            normalized_lines.append(line)

    normalized_text = "\n".join(normalized_lines).strip()
    unique_speakers = sorted({segment["speaker"] for segment in segments})

    metadata = {
        "parser": "transcript",
        "utterance_count": len(segments),
        "speaker_count": len(unique_speakers),
        "speakers": unique_speakers,
        "character_count": len(normalized_text),
        "segments_preview": segments[:5],
    }

    return normalized_text, metadata