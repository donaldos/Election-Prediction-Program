from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ingestion.base_registry import ComponentRegistry
from models.chunk import Chunk, ChunkWithEmbedding

logger = logging.getLogger(__name__)


class AbstractEmbedder(ABC):

    @abstractmethod
    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    def embed_query(self, text: str) -> list[float]:
        """단일 텍스트를 벡터로 변환. Retriever에서 질의 임베딩에 사용."""
        if not self.is_loaded:
            self.load()
        vectors = self._do_embed([text])
        return vectors[0]

    def embed(self, chunks: list[Chunk]) -> list[ChunkWithEmbedding]:
        if not chunks:
            logger.warning("[%s] 빈 청크 리스트 — 스킵", self.name)
            return []
        if not self.is_loaded:
            self.load()

        logger.info("[%s] 임베딩 시작 — %d개 청크", self.name, len(chunks))

        texts = [c.text for c in chunks]
        vectors = self._do_embed(texts)

        results = []
        for chunk, vector in zip(chunks, vectors):
            results.append(
                ChunkWithEmbedding(
                    **chunk.model_dump(),
                    embedding=vector,
                )
            )

        logger.info(
            "[%s] 임베딩 완료 — %d개 벡터 생성, 차원=%d",
            self.name, len(results), self.dimensions,
        )
        return results


EmbedderRegistry = ComponentRegistry(AbstractEmbedder, "Embedder")
