from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


EvalTaskType = Literal["retrieval", "qa", "summary"]


class EvalExample(BaseModel):
    example_id: str
    task_type: EvalTaskType
    query: str
    expected_source_files: list[str] = Field(default_factory=list)
    notes: str | None = None


def load_eval_examples(path: str) -> list[EvalExample]:
    eval_path = Path(path)

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

    raw_examples = json.loads(eval_path.read_text(encoding="utf-8"))

    if not isinstance(raw_examples, list):
        raise ValueError("Evaluation dataset must be a list of examples.")

    return [EvalExample(**example) for example in raw_examples]