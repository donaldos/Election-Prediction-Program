from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


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


class ChunkWithEmbedding(Chunk):
    """임베더 출력 단위. VectorDB 저장 직전 상태."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float]

    @property
    def metadata(self) -> dict:
        return self.model_dump(exclude={"id", "embedding"})
