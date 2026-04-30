from __future__ import annotations

import logging

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)


@VectorRepositoryRegistry.register("milvus_lite")
class MilvusLiteRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "election_chunks",
        db_path: str = "./milvus_lite.db",
        dimensions: int = 1536,
    ) -> None:
        self._collection_name = collection
        self._db_path = db_path
        self._dimensions = dimensions
        self._client = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "milvus_lite"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from pymilvus import MilvusClient

        self._client = MilvusClient(self._db_path)

        if not self._client.has_collection(self._collection_name):
            from pymilvus import CollectionSchema, DataType, FieldSchema

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dimensions),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="article_url", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="candidate", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="district_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunker_type", dtype=DataType.VARCHAR, max_length=256),
            ]
            schema = CollectionSchema(fields=fields)
            self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
            )
            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="COSINE")
            self._client.create_index(
                collection_name=self._collection_name,
                index_params=index_params,
            )
            logger.info("[%s] 컬렉션 생성: %s (dim=%d)", self.name, self._collection_name, self._dimensions)

        self._loaded = True
        logger.info("[%s] 로드 완료 — db=%s, collection=%s", self.name, self._db_path, self._collection_name)

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        data = []
        for c in chunks:
            row = {
                "id": c.id,
                "vector": c.embedding,
                "text": c.text,
                "article_url": c.article_url,
                "source": c.source,
                "title": c.title,
                "candidate": c.candidate,
                "district_id": c.district_id,
                "chunker_type": c.chunker_type,
            }
            data.append(row)
        self._client.upsert(collection_name=self._collection_name, data=data)
        return len(data)

    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]:
        filter_expr = ""
        if filters:
            parts = [f'{k} == "{v}"' for k, v in filters.items()]
            filter_expr = " and ".join(parts)

        results = self._client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=["text", "article_url", "source", "title", "candidate", "district_id", "chunker_type"],
        )

        items = []
        if results:
            for hit in results[0]:
                item = {"id": str(hit["id"]), "score": hit["distance"]}
                item.update(hit.get("entity", {}))
                items.append(item)
        return items

    def _do_delete(self, ids: list[str]) -> int:
        self._client.delete(collection_name=self._collection_name, ids=ids)
        return len(ids)

    def _do_count(self) -> int:
        stats = self._client.get_collection_stats(self._collection_name)
        return stats.get("row_count", 0)
