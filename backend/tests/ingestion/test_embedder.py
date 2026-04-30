from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.chunk import Chunk

SAMPLE_CHUNKS = [
    Chunk(
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
    ),
    Chunk(
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
    ),
]


# ── OpenAIEmbedder ────────────────────────────────────────


class TestOpenAIEmbedder:

    def test_embed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536, batch_size=100)

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]

        with patch("openai.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            MockClient.return_value = mock_client

            embedder.load()
            results = embedder.embed(SAMPLE_CHUNKS)

        assert len(results) == 2
        assert len(results[0].embedding) == 1536
        assert results[0].article_url == "https://example.com/1"
        assert results[1].candidate == "후보B"

    def test_embed_empty_chunks(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        embedder._loaded = True
        assert embedder.embed([]) == []

    def test_dimensions_property(self):
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        assert OpenAIEmbedder(dimensions=1536).dimensions == 1536
        assert OpenAIEmbedder(dimensions=3072).dimensions == 3072

    def test_name(self):
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        assert OpenAIEmbedder().name == "openai"

    def test_idempotent_load(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value = MagicMock()
            embedder.load()
            embedder.load()
            assert MockClient.call_count == 1

    def test_auto_load_on_embed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        assert not embedder.is_loaded

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("openai.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            MockClient.return_value = mock_client

            embedder.embed(SAMPLE_CHUNKS[:1])
            assert embedder.is_loaded

    def test_batch_processing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(batch_size=1)

        mock_response_1 = MagicMock()
        mock_response_1.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response_2 = MagicMock()
        mock_response_2.data = [MagicMock(embedding=[0.2] * 1536)]

        with patch("openai.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = [mock_response_1, mock_response_2]
            MockClient.return_value = mock_client

            embedder.load()
            results = embedder.embed(SAMPLE_CHUNKS)

        assert len(results) == 2
        assert mock_client.embeddings.create.call_count == 2

    def test_ada_002_no_dimensions(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-ada-002", dimensions=1536)

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("openai.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            MockClient.return_value = mock_client

            embedder.load()
            embedder.embed(SAMPLE_CHUNKS[:1])

            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert "dimensions" not in call_kwargs

    def test_metadata_preserved(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from ingestion.embedder.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("openai.OpenAI") as MockClient:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            MockClient.return_value = mock_client

            embedder.load()
            results = embedder.embed(SAMPLE_CHUNKS[:1])

        r = results[0]
        assert r.text == "평택을 선거구 여론조사 결과"
        assert r.source == "naver_news"
        assert r.chunker_type == "korean_paragraph"
        assert r.id is not None


# ── BGEM3Embedder ─────────────────────────────────────────


class TestBGEM3Embedder:

    def test_name_and_dimensions(self):
        from ingestion.embedder.bge import BGEM3Embedder

        embedder = BGEM3Embedder()
        assert embedder.name == "bge_m3"
        assert embedder.dimensions == 1024

    @pytest.mark.skip(reason="FlagEmbedding 모델 다운로드 필요 — 로컬에서만 실행")
    def test_load(self):
        from ingestion.embedder.bge import BGEM3Embedder

        embedder = BGEM3Embedder()
        embedder.load()
        assert embedder.is_loaded


# ── KoSimCSEEmbedder ──────────────────────────────────────


class TestKoSimCSEEmbedder:

    def test_name_and_dimensions(self):
        from ingestion.embedder.ko_simcse import KoSimCSEEmbedder

        embedder = KoSimCSEEmbedder()
        assert embedder.name == "ko_simcse"
        assert embedder.dimensions == 768

    @pytest.mark.skip(reason="sentence-transformers 모델 다운로드 필요 — 로컬에서만 실행")
    def test_load(self):
        from ingestion.embedder.ko_simcse import KoSimCSEEmbedder

        embedder = KoSimCSEEmbedder()
        embedder.load()
        assert embedder.is_loaded


# ── EmbedderRegistry ──────────────────────────────────────


class TestEmbedderRegistry:

    def test_all_registered(self):
        import ingestion.embedder  # noqa: F401
        from ingestion.embedder.base import EmbedderRegistry

        names = EmbedderRegistry.registered_names
        assert "openai" in names
        assert "bge_m3" in names
        assert "ko_simcse" in names

    def test_create_by_name(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        import ingestion.embedder  # noqa: F401
        from ingestion.embedder.base import EmbedderRegistry

        embedder = EmbedderRegistry.create(
            "openai", model="text-embedding-3-small", dimensions=1536, batch_size=100,
        )
        assert embedder.name == "openai"

    def test_create_unknown_raises(self):
        from ingestion.embedder.base import EmbedderRegistry

        with pytest.raises(ValueError, match="미등록"):
            EmbedderRegistry.create("nonexistent_embedder")
