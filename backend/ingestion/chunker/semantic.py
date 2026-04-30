from __future__ import annotations

import logging

from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("semantic")
class SemanticChunker(AbstractChunker):

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        breakpoint_threshold: float = 0.3,
        min_chunk_size: int = 100,
    ) -> None:
        self.model_name = model_name
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("[%s] loading model: %s ...", self.name, self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._loaded = True
        logger.info("[%s] model loaded", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        import kss
        import numpy as np

        sentences: list[str] = kss.split_sentences(text)
        if len(sentences) <= 1:
            return [self._make_chunk(text, metadata, 0)]

        embeddings = self._model.encode(sentences, normalize_embeddings=True)

        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        boundaries: list[int] = [
            i + 1
            for i, sim in enumerate(similarities)
            if sim < self.breakpoint_threshold
        ]

        groups: list[list[str]] = []
        prev = 0
        for boundary in boundaries:
            groups.append(sentences[prev:boundary])
            prev = boundary
        groups.append(sentences[prev:])

        merged: list[str] = []
        buffer = ""
        for group in groups:
            candidate = " ".join(group)
            if len(buffer) + len(candidate) < self.min_chunk_size:
                buffer += (" " + candidate) if buffer else candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = candidate
        if buffer:
            merged.append(buffer)

        return [self._make_chunk(t, metadata, i) for i, t in enumerate(merged)]
