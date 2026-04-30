from __future__ import annotations

import logging

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)


@VectorRepositoryRegistry.register("lancedb")
class LanceDBRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "election_chunks",
        db_path: str = "./lancedb_data",
        dimensions: int = 1536,
    ) -> None:
        self._collection_name = collection
        self._db_path = db_path
        self._dimensions = dimensions
        self._db = None
        self._table = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "lancedb"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import lancedb as ldb

        self._db = ldb.connect(self._db_path)

        existing = self._db.table_names()
        if self._collection_name in existing:
            self._table = self._db.open_table(self._collection_name)
        else:
            self._table = None

        self._loaded = True
        logger.info("[%s] 로드 완료 — db=%s, table=%s", self.name, self._db_path, self._collection_name)

    def _ensure_table(self, chunks: list[ChunkWithEmbedding]) -> None:
        if self._table is not None:
            return
        import pyarrow as pa

        data = self._chunks_to_records(chunks)
        self._table = self._db.create_table(self._collection_name, data=data)
        logger.info("[%s] 테이블 생성: %s", self.name, self._collection_name)

    @staticmethod
    def _chunks_to_records(chunks: list[ChunkWithEmbedding]) -> list[dict]:
        records = []
        for c in chunks:
            records.append({
                "id": c.id,
                "vector": c.embedding,
                "text": c.text,
                "article_url": c.article_url,
                "source": c.source,
                "title": c.title,
                "published_at": str(c.published_at),
                "candidate": c.candidate,
                "district_id": c.district_id,
                "chunker_type": c.chunker_type,
            })
        return records

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        if self._table is None:
            self._ensure_table(chunks)
            return len(chunks)

        records = self._chunks_to_records(chunks)
        import pyarrow as pa

        self._table.add(records)
        return len(records)

    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]:
        if self._table is None:
            return []

        query = self._table.search(query_vector).limit(top_k)

        if filters:
            parts = [f'{k} = "{v}"' for k, v in filters.items()]
            query = query.where(" AND ".join(parts))

        results = query.to_list()
        items = []
        for row in results:
            item = {
                "id": row.get("id", ""),
                "score": 1.0 - row.get("_distance", 0),
                "text": row.get("text", ""),
                "article_url": row.get("article_url", ""),
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "candidate": row.get("candidate", ""),
                "district_id": row.get("district_id", ""),
            }
            items.append(item)
        return items

    def _do_delete(self, ids: list[str]) -> int:
        if self._table is None:
            return 0
        id_list = ", ".join(f'"{i}"' for i in ids)
        self._table.delete(f"id IN ({id_list})")
        return len(ids)

    def _do_count(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()
