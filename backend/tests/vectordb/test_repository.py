from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry


SAMPLE_CHUNKS = [
    ChunkWithEmbedding(
        id="chunk-001",
        text="평택을 선거구 여론조사 결과",
        chunk_index=0,
        char_count=14,
        article_url="https://example.com/1",
        source="naver_news",
        title="판세 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보A",
        district_id="pyeongtaek_b",
        chunker_type="korean_paragraph",
        embedding=[0.1] * 1536,
    ),
    ChunkWithEmbedding(
        id="chunk-002",
        text="부산북구갑 후보 간 격차 분석",
        chunk_index=1,
        char_count=15,
        article_url="https://example.com/2",
        source="naver_news",
        title="격차 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보B",
        district_id="busan_bukgu_gap",
        chunker_type="korean_paragraph",
        embedding=[0.2] * 1536,
    ),
]


# ── AbstractVectorRepository ─────────────────────────────


class TestAbstractVectorRepository:

    def test_upsert_empty_returns_zero(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        result = AbstractVectorRepository.upsert(repo, [])
        assert result == 0

    def test_upsert_calls_do_upsert(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = True
        repo._do_upsert.return_value = 2

        result = AbstractVectorRepository.upsert(repo, SAMPLE_CHUNKS)
        assert result == 2
        repo._do_upsert.assert_called_once_with(SAMPLE_CHUNKS)

    def test_upsert_auto_loads(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = False
        repo._do_upsert.return_value = 1

        AbstractVectorRepository.upsert(repo, SAMPLE_CHUNKS[:1])
        repo.load.assert_called_once()

    def test_search_calls_do_search(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = True
        repo._do_search.return_value = [{"id": "1", "score": 0.9}]

        results = AbstractVectorRepository.search(repo, [0.1] * 1536, top_k=5)
        assert len(results) == 1
        repo._do_search.assert_called_once_with([0.1] * 1536, 5, None)

    def test_search_with_filters(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = True
        repo._do_search.return_value = []

        filters = {"district_id": "pyeongtaek_b"}
        AbstractVectorRepository.search(repo, [0.1] * 1536, top_k=3, filters=filters)
        repo._do_search.assert_called_once_with([0.1] * 1536, 3, filters)

    def test_delete_empty_returns_zero(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        result = AbstractVectorRepository.delete(repo, [])
        assert result == 0

    def test_delete_calls_do_delete(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = True
        repo._do_delete.return_value = 2

        result = AbstractVectorRepository.delete(repo, ["id1", "id2"])
        assert result == 2

    def test_count_auto_loads(self):
        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = False
        repo._do_count.return_value = 42

        result = AbstractVectorRepository.count(repo)
        assert result == 42
        repo.load.assert_called_once()

    @patch("vectordb.base.date")
    def test_delete_older_than_calls_find_and_delete(self, mock_date):
        mock_date.today.return_value = date(2026, 4, 30)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = True
        repo._find_ids_older_than.return_value = ["old-1", "old-2"]
        repo._do_delete.return_value = 2

        result = AbstractVectorRepository.delete_older_than(repo, 30)

        assert result == 2
        repo._find_ids_older_than.assert_called_once()
        repo._do_delete.assert_called_once_with(["old-1", "old-2"])

    @patch("vectordb.base.date")
    def test_delete_older_than_no_targets(self, mock_date):
        mock_date.today.return_value = date(2026, 4, 30)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        repo = MagicMock(spec=AbstractVectorRepository)
        repo.name = "mock"
        repo.is_loaded = True
        repo._find_ids_older_than.return_value = []

        result = AbstractVectorRepository.delete_older_than(repo, 30)

        assert result == 0
        repo._do_delete.assert_not_called()

    def test_default_find_ids_older_than_returns_empty(self):
        result = AbstractVectorRepository._find_ids_older_than(MagicMock(), "2026-01-01T00:00:00")
        assert result == []


# ── VectorRepositoryRegistry ─────────────────────────────


class TestVectorRepositoryRegistry:

    def test_all_registered(self):
        import vectordb  # noqa: F401

        names = VectorRepositoryRegistry.registered_names
        assert "qdrant" in names
        assert "chroma" in names
        assert "milvus_lite" in names
        assert "lancedb" in names
        assert "weaviate" in names
        assert "pgvector" in names

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="미등록"):
            VectorRepositoryRegistry.create("nonexistent_db")

    def test_create_qdrant(self):
        import vectordb  # noqa: F401

        repo = VectorRepositoryRegistry.create("qdrant", collection="test", host="localhost", port=6333)
        assert repo.name == "qdrant"
        assert not repo.is_loaded

    def test_create_chroma(self):
        import vectordb  # noqa: F401

        repo = VectorRepositoryRegistry.create("chroma", collection="test", persist_dir="/tmp/test_chroma")
        assert repo.name == "chroma"
        assert not repo.is_loaded

    def test_create_milvus_lite(self):
        import vectordb  # noqa: F401

        repo = VectorRepositoryRegistry.create("milvus_lite", collection="test", db_path="./test.db")
        assert repo.name == "milvus_lite"
        assert not repo.is_loaded

    def test_create_lancedb(self):
        import vectordb  # noqa: F401

        repo = VectorRepositoryRegistry.create("lancedb", collection="test", db_path="./test_lance")
        assert repo.name == "lancedb"
        assert not repo.is_loaded

    def test_create_weaviate(self):
        import vectordb  # noqa: F401

        repo = VectorRepositoryRegistry.create("weaviate", collection="TestCol", host="localhost")
        assert repo.name == "weaviate"
        assert not repo.is_loaded

    def test_create_pgvector(self):
        import vectordb  # noqa: F401

        repo = VectorRepositoryRegistry.create("pgvector", collection="test", dsn="postgresql://localhost/test")
        assert repo.name == "pgvector"
        assert not repo.is_loaded


# ── QdrantRepository ─────────────────────────────────────


class TestQdrantRepository:

    def test_name(self):
        from vectordb.qdrant_repo import QdrantRepository

        assert QdrantRepository().name == "qdrant"

    def test_idempotent_load(self):
        from vectordb.qdrant_repo import QdrantRepository

        repo = QdrantRepository()

        mock_qdrant_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="election_chunks")]
        mock_qdrant_client.get_collections.return_value = mock_collections
        MockClientClass = MagicMock(return_value=mock_qdrant_client)

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(QdrantClient=MockClientClass), "qdrant_client.models": MagicMock()}):
            repo.load()
            repo.load()
            assert MockClientClass.call_count == 1

    def test_upsert(self):
        from vectordb.qdrant_repo import QdrantRepository

        repo = QdrantRepository()
        repo._loaded = True
        repo._client = MagicMock()

        mock_point_struct = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(), "qdrant_client.models": MagicMock(PointStruct=mock_point_struct)}):
            count = repo.upsert(SAMPLE_CHUNKS)
            assert count == 2
            repo._client.upsert.assert_called_once()

    def test_search(self):
        from vectordb.qdrant_repo import QdrantRepository

        repo = QdrantRepository()
        repo._loaded = True

        mock_result = MagicMock()
        mock_result.id = "chunk-001"
        mock_result.score = 0.95
        mock_result.payload = {"text": "test", "candidate": "후보A"}

        repo._client = MagicMock()
        repo._client.search.return_value = [mock_result]

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(), "qdrant_client.models": MagicMock()}):
            results = repo.search([0.1] * 1536, top_k=5)
            assert len(results) == 1
            assert results[0]["score"] == 0.95

    def test_count(self):
        from vectordb.qdrant_repo import QdrantRepository

        repo = QdrantRepository()
        repo._loaded = True
        repo._client = MagicMock()
        mock_info = MagicMock()
        mock_info.points_count = 42
        repo._client.get_collection.return_value = mock_info

        assert repo.count() == 42


# ── ChromaRepository ─────────────────────────────────────


class TestChromaRepository:

    def test_name(self):
        from vectordb.chroma_repo import ChromaRepository

        assert ChromaRepository().name == "chroma"

    def test_load_and_upsert(self, tmp_path):
        from vectordb.chroma_repo import ChromaRepository

        repo = ChromaRepository(collection="test", persist_dir=str(tmp_path / "chroma_test"))

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        MockPersistentClient = MagicMock(return_value=mock_client)

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient = MockPersistentClient

        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "chromadb.config": MagicMock()}):
            repo.load()
            assert repo.is_loaded

            count = repo.upsert(SAMPLE_CHUNKS)
            assert count == 2
            mock_collection.upsert.assert_called_once()

    def test_search(self):
        from vectordb.chroma_repo import ChromaRepository

        repo = ChromaRepository()
        repo._loaded = True
        repo._collection = MagicMock()
        repo._collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "metadatas": [[{"candidate": "후보A"}]],
            "documents": [["test text"]],
        }

        results = repo.search([0.1] * 1536, top_k=5)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.9)

    def test_count(self):
        from vectordb.chroma_repo import ChromaRepository

        repo = ChromaRepository()
        repo._loaded = True
        repo._collection = MagicMock()
        repo._collection.count.return_value = 10

        assert repo.count() == 10

    def test_find_ids_older_than(self):
        from vectordb.chroma_repo import ChromaRepository

        repo = ChromaRepository()
        repo._loaded = True
        repo._collection = MagicMock()
        repo._collection.get.return_value = {"ids": ["old-1", "old-2"]}

        ids = repo._find_ids_older_than("2026-04-01T00:00:00")

        assert ids == ["old-1", "old-2"]
        call_args = repo._collection.get.call_args
        where = call_args[1]["where"]
        assert "published_at_ts" in where
        assert "$lt" in where["published_at_ts"]

    def test_find_ids_older_than_empty(self):
        from vectordb.chroma_repo import ChromaRepository

        repo = ChromaRepository()
        repo._loaded = True
        repo._collection = MagicMock()
        repo._collection.get.return_value = {"ids": []}

        ids = repo._find_ids_older_than("2026-04-01T00:00:00")

        assert ids == []


# ── MilvusLiteRepository (skip — pymilvus 설치 필요) ─────


class TestMilvusLiteRepository:

    def test_name(self):
        from vectordb.milvus_repo import MilvusLiteRepository

        assert MilvusLiteRepository().name == "milvus_lite"

    @pytest.mark.skip(reason="pymilvus 설치 필요 — 로컬에서만 실행")
    def test_load(self):
        from vectordb.milvus_repo import MilvusLiteRepository

        repo = MilvusLiteRepository(db_path="./test_milvus.db")
        repo.load()
        assert repo.is_loaded


# ── LanceDBRepository (skip — lancedb 설치 필요) ─────────


class TestLanceDBRepository:

    def test_name(self):
        from vectordb.lancedb_repo import LanceDBRepository

        assert LanceDBRepository().name == "lancedb"

    @pytest.mark.skip(reason="lancedb 설치 필요 — 로컬에서만 실행")
    def test_load(self):
        from vectordb.lancedb_repo import LanceDBRepository

        repo = LanceDBRepository(db_path="./test_lance")
        repo.load()
        assert repo.is_loaded


# ── WeaviateRepository (skip — weaviate 서버 필요) ───────


class TestWeaviateRepository:

    def test_name(self):
        from vectordb.weaviate_repo import WeaviateRepository

        assert WeaviateRepository().name == "weaviate"

    @pytest.mark.skip(reason="weaviate 서버 필요 — 로컬에서만 실행")
    def test_load(self):
        from vectordb.weaviate_repo import WeaviateRepository

        repo = WeaviateRepository()
        repo.load()
        assert repo.is_loaded


# ── PgvectorRepository (skip — PostgreSQL 필요) ──────────


class TestPgvectorRepository:

    def test_name(self):
        from vectordb.pgvector_repo import PgvectorRepository

        assert PgvectorRepository().name == "pgvector"

    @pytest.mark.skip(reason="PostgreSQL + pgvector 확장 필요 — 로컬에서만 실행")
    def test_load(self):
        from vectordb.pgvector_repo import PgvectorRepository

        repo = PgvectorRepository()
        repo.load()
        assert repo.is_loaded


# ── PineconeRepository ──────────────────────────────────


class TestPineconeRepository:

    def test_name(self):
        from vectordb.pinecone_repo import PineconeRepository

        assert PineconeRepository(api_key="test-key").name == "pinecone"

    def test_registry_lookup(self):
        repo = VectorRepositoryRegistry.create(
            "pinecone", collection="test", api_key="test-key",
        )
        assert repo.name == "pinecone"

    def test_load_creates_index_if_missing(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key", index_name="test-idx")

        mock_index = MagicMock()
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = []
        mock_pc.Index.return_value = mock_index

        with patch.dict("sys.modules", {"pinecone": MagicMock(Pinecone=MagicMock(return_value=mock_pc), ServerlessSpec=MagicMock())}):
            repo.load()
            mock_pc.create_index.assert_called_once()
            assert repo.is_loaded

    def test_load_skips_create_if_exists(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key", index_name="test-idx")

        mock_existing = MagicMock()
        mock_existing.name = "test-idx"
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = [mock_existing]
        mock_pc.Index.return_value = MagicMock()

        with patch.dict("sys.modules", {"pinecone": MagicMock(Pinecone=MagicMock(return_value=mock_pc), ServerlessSpec=MagicMock())}):
            repo.load()
            mock_pc.create_index.assert_not_called()

    def test_upsert(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key")
        repo._loaded = True
        repo._index = MagicMock()

        count = repo.upsert(SAMPLE_CHUNKS)
        assert count == 2
        repo._index.upsert.assert_called_once()

    def test_search(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key")
        repo._loaded = True
        repo._index = MagicMock()
        repo._index.query.return_value = {
            "matches": [
                {"id": "chunk-001", "score": 0.95, "metadata": {"text": "test", "candidate": "후보A"}},
            ]
        }

        results = repo.search([0.1] * 1536, top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "chunk-001"
        assert results[0]["score"] == 0.95
        assert results[0]["candidate"] == "후보A"
        assert "text" not in results[0].get("metadata", {})

    def test_search_with_filters(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key")
        repo._loaded = True
        repo._index = MagicMock()
        repo._index.query.return_value = {"matches": []}

        repo.search([0.1] * 1536, top_k=5, filters={"district_id": "pyeongtaek_b"})
        call_kwargs = repo._index.query.call_args[1]
        assert call_kwargs["filter"] == {"district_id": {"$eq": "pyeongtaek_b"}}

    def test_count(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key")
        repo._loaded = True
        repo._index = MagicMock()
        repo._index.describe_index_stats.return_value = {"total_vector_count": 42}

        assert repo.count() == 42

    def test_delete(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository(api_key="test-key")
        repo._loaded = True
        repo._index = MagicMock()

        count = repo.delete(["id-1", "id-2"])
        assert count == 2
        repo._index.delete.assert_called_once()

    @pytest.mark.skip(reason="Pinecone 클라우드 계정 + API 키 필요 — 로컬에서만 실행")
    def test_load_real(self):
        from vectordb.pinecone_repo import PineconeRepository

        repo = PineconeRepository()
        repo.load()
        assert repo.is_loaded
