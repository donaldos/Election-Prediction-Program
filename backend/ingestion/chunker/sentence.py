from __future__ import annotations

import logging

from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("sentence")
class SentenceChunker(AbstractChunker):

    def __init__(self, sentences_per_chunk: int = 5) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self._kss = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "sentence"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import kss

        self._kss = kss
        self._loaded = True
        logger.info("[%s] kss loaded", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        sentences: list[str] = self._kss.split_sentences(text)
        chunks: list[Chunk] = []

        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = sentences[i : i + self.sentences_per_chunk]
            chunks.append(self._make_chunk(" ".join(group), metadata, len(chunks)))

        return chunks
