from __future__ import annotations

import hashlib
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """청커 출력 단위. 임베딩 이전 상태."""

    text: str
    chunk_index: int
    char_count: int

    article_url: str
    source: str
    title: str
    published_at: datetime
    candidate: str
    district_id: str

    chunker_type: str

    pollster: str = ""
    poll_survey_date: str = ""
    sample_size: int = 0
    margin_of_error: float = 0.0


def _deterministic_id(article_url: str, chunk_index: int) -> str:
    key = f"{article_url}::chunk::{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class ChunkWithEmbedding(Chunk):
    """임베더 출력 단위. VectorDB 저장 직전 상태."""

    id: str = Field(default="")
    embedding: list[float]

    @model_validator(mode="after")
    def _set_deterministic_id(self) -> ChunkWithEmbedding:
        if not self.id:
            self.id = _deterministic_id(self.article_url, self.chunk_index)
        return self

    @property
    def metadata(self) -> dict:
        return self.model_dump(exclude={"id", "embedding"})
