from __future__ import annotations

import logging

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)


@VectorRepositoryRegistry.register("chroma")
class ChromaRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "election_chunks",
        persist_dir: str = ".chroma",
    ) -> None:
        self._collection_name = collection
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "chroma"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._loaded = True
        logger.info("[%s] 로드 완료 — persist=%s, collection=%s", self.name, self._persist_dir, self._collection_name)

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        ids = [c.id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = []
        documents = []
        for c in chunks:
            meta = c.metadata
            meta["published_at"] = str(meta["published_at"])
            metadatas.append(meta)
            documents.append(c.text)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(ids)

    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]:
        where = None
        if filters:
            if len(filters) == 1:
                where = filters
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                item = {
                    "id": doc_id,
                    "score": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                }
                if results["metadatas"]:
                    item.update(results["metadatas"][0][i])
                if results["documents"]:
                    item["text"] = results["documents"][0][i]
                items.append(item)
        return items

    def _do_delete(self, ids: list[str]) -> int:
        self._collection.delete(ids=ids)
        return len(ids)

    def _do_count(self) -> int:
        return self._collection.count()
