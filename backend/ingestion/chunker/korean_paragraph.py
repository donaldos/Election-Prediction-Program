from __future__ import annotations

import logging

from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("korean_paragraph")
class KoreanParagraphChunker(AbstractChunker):

    def __init__(self, chunk_size: int = 400, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._loaded = False

    @property
    def name(self) -> str:
        return "korean_paragraph"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        logger.info("[%s] loaded (no external deps)", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        buffer = ""

        for para in paragraphs:
            if len(buffer) + len(para) <= self.chunk_size:
                buffer += ("\n\n" + para) if buffer else para
            else:
                if buffer:
                    chunks.append(self._make_chunk(buffer, metadata, len(chunks)))
                tail = buffer[-self.overlap :] if len(buffer) > self.overlap else buffer
                buffer = (tail + "\n\n" + para) if tail else para

        if buffer:
            chunks.append(self._make_chunk(buffer, metadata, len(chunks)))

        return chunks
