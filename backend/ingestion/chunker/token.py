from __future__ import annotations

import logging

from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("token")
class TokenChunker(AbstractChunker):

    def __init__(
        self,
        tokens_per_chunk: int = 256,
        overlap_tokens: int = 32,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name
        self._enc = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "token"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import tiktoken

        self._enc = tiktoken.get_encoding(self.encoding_name)
        self._loaded = True
        logger.info("[%s] tiktoken(%s) loaded", self.name, self.encoding_name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        token_ids: list[int] = self._enc.encode(text)
        chunks: list[Chunk] = []
        start = 0

        while start < len(token_ids):
            end = min(start + self.tokens_per_chunk, len(token_ids))
            window = token_ids[start:end]
            chunk_text = self._enc.decode(window)
            chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))
            if end == len(token_ids):
                break
            start += self.tokens_per_chunk - self.overlap_tokens

        return chunks
