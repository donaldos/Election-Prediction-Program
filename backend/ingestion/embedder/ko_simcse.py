from __future__ import annotations

import logging

from ingestion.embedder.base import AbstractEmbedder, EmbedderRegistry

logger = logging.getLogger(__name__)


@EmbedderRegistry.register("ko_simcse")
class KoSimCSEEmbedder(AbstractEmbedder):

    def __init__(
        self,
        model_name: str = "BM-K/KoSimCSE-roberta",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "ko_simcse"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimensions(self) -> int:
        return 768

    def load(self) -> None:
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("[%s] loading model: %s ...", self.name, self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._loaded = True
        logger.info("[%s] model loaded", self.name)

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug("[%s] 로컬 추론 — batch %d~%d", self.name, i, i + len(batch))
            vectors = self._model.encode(batch, normalize_embeddings=True)
            all_vectors.extend(vectors.tolist())

        return all_vectors
