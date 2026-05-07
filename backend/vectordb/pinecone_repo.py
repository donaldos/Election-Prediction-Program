from __future__ import annotations

import logging
import os
from datetime import datetime

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100


@VectorRepositoryRegistry.register("pinecone")
class PineconeRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "election_chunks",
        index_name: str = "election-radar",
        dimensions: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: str | None = None,
    ) -> None:
        self._collection = collection
        self._index_name = index_name
        self._dimensions = dimensions
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self._index = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "pinecone"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from pinecone import Pinecone, ServerlessSpec

        if not self._api_key:
            raise ValueError("PINECONE_API_KEY가 설정되지 않았습니다.")

        pc = Pinecone(api_key=self._api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if self._index_name not in existing:
            pc.create_index(
                name=self._index_name,
                dimension=self._dimensions,
                metric=self._metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
            logger.info(
                "[%s] 인덱스 생성: %s (dim=%d, metric=%s)",
                self.name, self._index_name, self._dimensions, self._metric,
            )

        self._index = pc.Index(self._index_name)
        self._loaded = True
        logger.info("[%s] 연결 완료 — index=%s", self.name, self._index_name)

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        vectors = []
        for c in chunks:
            meta = dict(c.metadata)
            pub = meta.get("published_at")
            if isinstance(pub, datetime):
                meta["published_at_ts"] = pub.timestamp()
                meta["published_at"] = pub.isoformat()
            elif isinstance(pub, str):
                meta["published_at_ts"] = datetime.fromisoformat(pub).timestamp()
            else:
                meta["published_at_ts"] = 0.0
                meta["published_at"] = str(pub) if pub else ""

            meta["text"] = c.text
            vectors.append({
                "id": c.id,
                "values": c.embedding,
                "metadata": meta,
            })

        total = 0
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch)
            total += len(batch)

        return total

    def _do_search(
        self, query_vector: list[float], top_k: int, filters: dict | None,
    ) -> list[dict]:
        pine_filter = None
        if filters:
            pine_filter = {k: {"$eq": v} for k, v in filters.items()}

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            filter=pine_filter,
            include_metadata=True,
        )

        items = []
        for match in results.get("matches", []):
            meta = dict(match.get("metadata", {}))
            text = meta.pop("text", "")
            items.append({
                "id": match["id"],
                "score": match["score"],
                "text": text,
                **meta,
            })
        return items

    def _do_delete(self, ids: list[str]) -> int:
        for i in range(0, len(ids), UPSERT_BATCH_SIZE):
            batch = ids[i : i + UPSERT_BATCH_SIZE]
            self._index.delete(ids=batch)
        return len(ids)

    def _do_count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def _find_ids_older_than(self, cutoff_iso: str) -> list[str]:
        cutoff_ts = datetime.fromisoformat(cutoff_iso).timestamp()

        results = self._index.query(
            vector=[0.0] * self._dimensions,
            top_k=10000,
            filter={"published_at_ts": {"$lt": cutoff_ts}},
            include_metadata=False,
        )

        return [m["id"] for m in results.get("matches", [])]
