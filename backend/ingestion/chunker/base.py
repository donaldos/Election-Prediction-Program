from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ingestion.base_registry import ComponentRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


class AbstractChunker(ABC):

    @abstractmethod
    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
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

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        if not text.strip():
            logger.warning("[%s] 빈 텍스트 입력 — 스킵", self.name)
            return []
        if not self.is_loaded:
            self.load()

        title = metadata.get("title", "")
        logger.info("[%s] 청킹 시작 — 입력 %d자, 제목='%s'", self.name, len(text), title[:30])

        chunks = self._do_chunk(text, metadata)

        sizes = [c.char_count for c in chunks]
        logger.info(
            "[%s] 청킹 완료 — %d개 청크 생성, 평균 %d자, 최소 %d자, 최대 %d자",
            self.name, len(chunks),
            sum(sizes) // len(sizes) if sizes else 0,
            min(sizes) if sizes else 0,
            max(sizes) if sizes else 0,
        )
        for c in chunks:
            logger.debug("[%s] chunk[%d] %d자: '%s...'", self.name, c.chunk_index, c.char_count, c.text[:40])

        return chunks

    def _make_chunk(self, text: str, metadata: dict, idx: int) -> Chunk:
        return Chunk(
            text=text.strip(),
            chunk_index=idx,
            char_count=len(text.strip()),
            chunker_type=self.name,
            **metadata,
        )


ChunkerRegistry = ComponentRegistry(AbstractChunker, "Chunker")
