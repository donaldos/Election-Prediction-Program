from __future__ import annotations

import logging
import os

from ingestion.embedder.base import AbstractEmbedder, EmbedderRegistry

logger = logging.getLogger(__name__)


@EmbedderRegistry.register("openai")
class OpenAIEmbedder(AbstractEmbedder):

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
    ) -> None:
        self._model_name = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "openai"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def load(self) -> None:
        if self._loaded:
            return
        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._loaded = True
        logger.info(
            "[%s] client initialized — model=%s, dim=%d",
            self.name, self._model_name, self._dimensions,
        )

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug("[%s] API 호출 — batch %d~%d", self.name, i, i + len(batch))

            kwargs: dict = {
                "model": self._model_name,
                "input": batch,
            }
            if self._model_name != "text-embedding-ada-002":
                kwargs["dimensions"] = self._dimensions

            response = self._client.embeddings.create(**kwargs)
            batch_vectors = [item.embedding for item in response.data]
            all_vectors.extend(batch_vectors)

        return all_vectors
