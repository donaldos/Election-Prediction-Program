from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ingestion.base_registry import ComponentRegistry
from models.chunk import ChunkWithEmbedding

logger = logging.getLogger(__name__)


class AbstractVectorRepository(ABC):

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

    def upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        if not chunks:
            logger.warning("[%s] 빈 벡터 리스트 — 스킵", self.name)
            return 0
        if not self.is_loaded:
            self.load()

        logger.info("[%s] upsert 시작 — %d개 벡터", self.name, len(chunks))
        count = self._do_upsert(chunks)
        logger.info("[%s] upsert 완료 — %d개 저장", self.name, count)
        return count

    @abstractmethod
    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        if not self.is_loaded:
            self.load()

        logger.info("[%s] 검색 시작 — top_k=%d", self.name, top_k)
        results = self._do_search(query_vector, top_k, filters)
        logger.info("[%s] 검색 완료 — %d개 결과", self.name, len(results))
        return results

    @abstractmethod
    def _do_search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None,
    ) -> list[dict]:
        ...

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        if not self.is_loaded:
            self.load()

        logger.info("[%s] 삭제 시작 — %d개", self.name, len(ids))
        count = self._do_delete(ids)
        logger.info("[%s] 삭제 완료 — %d개", self.name, count)
        return count

    @abstractmethod
    def _do_delete(self, ids: list[str]) -> int:
        ...

    def count(self) -> int:
        if not self.is_loaded:
            self.load()
        return self._do_count()

    @abstractmethod
    def _do_count(self) -> int:
        ...


VectorRepositoryRegistry = ComponentRegistry(AbstractVectorRepository, "VectorRepository")
