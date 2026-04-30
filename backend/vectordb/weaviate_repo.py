from __future__ import annotations

import logging

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)


@VectorRepositoryRegistry.register("weaviate")
class WeaviateRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "ElectionChunks",
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        dimensions: int = 1536,
    ) -> None:
        self._collection_name = collection
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._dimensions = dimensions
        self._client = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "weaviate"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import weaviate
        import weaviate.classes.config as wvc

        self._client = weaviate.connect_to_local(
            host=self._host,
            port=self._port,
            grpc_port=self._grpc_port,
        )

        if not self._client.collections.exists(self._collection_name):
            self._client.collections.create(
                name=self._collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                properties=[
                    wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="article_url", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="source", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="published_at", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="candidate", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="district_id", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="chunker_type", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="chunk_id", data_type=wvc.DataType.TEXT),
                ],
            )
            logger.info("[%s] 컬렉션 생성: %s", self.name, self._collection_name)

        self._loaded = True
        logger.info("[%s] 연결 완료 — %s:%d/%s", self.name, self._host, self._port, self._collection_name)

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        import weaviate.classes.data as wvd

        collection = self._client.collections.get(self._collection_name)
        with collection.batch.dynamic() as batch:
            for c in chunks:
                batch.add_object(
                    properties={
                        "text": c.text,
                        "article_url": c.article_url,
                        "source": c.source,
                        "title": c.title,
                        "published_at": str(c.published_at),
                        "candidate": c.candidate,
                        "district_id": c.district_id,
                        "chunker_type": c.chunker_type,
                        "chunk_id": c.id,
                    },
                    vector=c.embedding,
                    uuid=wvd.DataObject.generate_deterministic_id(c.id),
                )
        return len(chunks)

    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]:
        import weaviate.classes.query as wvq

        collection = self._client.collections.get(self._collection_name)

        wv_filters = None
        if filters:
            filter_list = []
            for k, v in filters.items():
                filter_list.append(wvq.Filter.by_property(k).equal(v))
            wv_filters = filter_list[0] if len(filter_list) == 1 else wvq.Filter.all_of(filter_list)

        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=wv_filters,
            return_metadata=wvq.MetadataQuery(distance=True),
        )

        items = []
        for obj in results.objects:
            item = {
                "id": obj.properties.get("chunk_id", str(obj.uuid)),
                "score": 1.0 - (obj.metadata.distance or 0),
            }
            item.update(obj.properties)
            items.append(item)
        return items

    def _do_delete(self, ids: list[str]) -> int:
        import weaviate.classes.data as wvd

        collection = self._client.collections.get(self._collection_name)
        for chunk_id in ids:
            uuid = wvd.DataObject.generate_deterministic_id(chunk_id)
            collection.data.delete_by_id(uuid)
        return len(ids)

    def _do_count(self) -> int:
        collection = self._client.collections.get(self._collection_name)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count
