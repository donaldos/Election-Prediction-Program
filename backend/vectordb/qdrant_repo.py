from __future__ import annotations

import logging

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)


@VectorRepositoryRegistry.register("qdrant")
class QdrantRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "election_chunks",
        host: str = "localhost",
        port: int = 6333,
        dimensions: int = 1536,
    ) -> None:
        self._collection = collection
        self._host = host
        self._port = port
        self._dimensions = dimensions
        self._client = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "qdrant"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(host=self._host, port=self._port)

        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("[%s] 컬렉션 생성: %s (dim=%d)", self.name, self._collection, self._dimensions)

        self._loaded = True
        logger.info("[%s] 연결 완료 — %s:%d/%s", self.name, self._host, self._port, self._collection)

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload=chunk.metadata,
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                **r.payload,
            }
            for r in results
        ]

    def _do_delete(self, ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    def _do_count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count

    def _find_ids_older_than(self, cutoff_iso: str) -> list[str]:
        from qdrant_client.models import Filter, FieldCondition, Range

        results = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="published_at", range=Range(lt=cutoff_iso)),
            ]),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        points, _ = results
        return [str(p.id) for p in points]
