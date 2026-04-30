from __future__ import annotations

import logging

from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@ChunkerRegistry.register("recursive")
class RecursiveChunker(AbstractChunker):

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self._loaded = False

    @property
    def name(self) -> str:
        return "recursive"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        logger.info("[%s] loaded (no external deps)", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        raw_chunks = self._split(text, self.separators)

        result: list[str] = []
        for i, c in enumerate(raw_chunks):
            if i == 0 or not result:
                result.append(c)
            else:
                tail = result[-1][-self.overlap :] if len(result[-1]) > self.overlap else result[-1]
                result.append(tail + c)

        return [self._make_chunk(t, metadata, i) for i, t in enumerate(result) if t.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining = separators[1:]

        if len(text) <= self.chunk_size:
            return [text]

        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        buffer = ""

        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, remaining))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            chunks.append(buffer)

        return chunks
