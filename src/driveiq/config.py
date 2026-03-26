from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_data_dir: str
    processed_data_dir: str
    index_data_dir: str
    eval_data_dir: str
    artifacts_dir: str


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    app_name: str
    environment: str = "dev"
    log_level: str = "INFO"
    paths: PathsConfig
    api: ApiConfig


class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 75


class RetrievalSettings(BaseModel):
    top_k: int = 5
    score_function: str = "cosine"


class EmbeddingConfig(BaseModel):
    provider: str = "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"


class RetrievalConfig(BaseModel):
    chunking: ChunkingConfig
    retrieval: RetrievalSettings
    embedding: EmbeddingConfig


class EvaluationSettings(BaseModel):
    enabled: bool = True
    sample_size: int = 5
    write_reports: bool = True


class EvalConfig(BaseModel):
    evaluation: EvaluationSettings


class Settings(BaseModel):
    app: AppConfig
    retrieval: RetrievalConfig
    eval: EvalConfig


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")

    return data


@lru_cache(maxsize=1)
def get_settings(config_dir: str = "configs") -> Settings:
    config_path = Path(config_dir)

    app_data = _load_yaml_file(config_path / "app.yaml")
    retrieval_data = _load_yaml_file(config_path / "retrieval.yaml")
    eval_data = _load_yaml_file(config_path / "eval.yaml")

    return Settings(
        app=AppConfig(**app_data),
        retrieval=RetrievalConfig(**retrieval_data),
        eval=EvalConfig(**eval_data),
    )


if __name__ == "__main__":
    settings = get_settings()
    print(settings.model_dump())