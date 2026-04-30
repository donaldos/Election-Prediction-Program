from __future__ import annotations

import logging

from ingestion.embedder.base import AbstractEmbedder, EmbedderRegistry

logger = logging.getLogger(__name__)


@EmbedderRegistry.register("bge_m3")
class BGEM3Embedder(AbstractEmbedder):

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "bge_m3"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimensions(self) -> int:
        return 1024

    def load(self) -> None:
        if self._loaded:
            return
        from FlagEmbedding import BGEM3FlagModel

        logger.info("[%s] loading model: %s ...", self.name, self._model_name)
        self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
        self._loaded = True
        logger.info("[%s] model loaded", self.name)

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug("[%s] 로컬 추론 — batch %d~%d", self.name, i, i + len(batch))
            result = self._model.encode(batch)
            vectors = result["dense_vecs"].tolist()
            all_vectors.extend(vectors)

        return all_vectors
