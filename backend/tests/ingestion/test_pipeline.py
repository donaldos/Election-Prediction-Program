from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ingestion.pipeline import IngestionPipeline
from models.article import RawArticle
from models.chunk import Chunk, ChunkWithEmbedding


SAMPLE_CONFIG = {
    "districts": [
        {
            "id": "pyeongtaek_b",
            "name": "평택을",
            "candidates": [
                {"name": "후보A", "party": "A당", "keywords": ["후보A", "평택을"]},
            ],
        },
    ],
    "scrapers": {
        "naver": {"type": "naver", "params": {"max_articles_per_run": 10, "request_delay_sec": 0, "lookback_days": 1}},
        "political": {"type": "political", "params": {"urls": [], "max_articles_per_run": 10, "request_delay_sec": 0, "lookback_days": 1}},
    },
    "chunker": {"type": "korean_paragraph", "params": {"chunk_size": 200, "overlap": 0}},
    "embedder": {"type": "openai", "params": {"model": "text-embedding-3-small", "dimensions": 1536, "batch_size": 100}},
    "vectordb": {"type": "chroma", "collection": "test_chunks", "params": {"persist_dir": "/tmp/test_chroma"}},
}

SAMPLE_ARTICLES = [
    RawArticle(
        url="https://example.com/1",
        source="naver_news",
        title="평택을 판세 분석",
        body="평택을 선거구의 여론조사 결과가 발표되었다. " * 10,
        published_at=datetime(2026, 5, 1),
        candidate="후보A",
        district_id="pyeongtaek_b",
    ),
]

SAMPLE_CHUNKS = [
    Chunk(
        text="평택을 선거구의 여론조사 결과가 발표되었다.",
        chunk_index=0,
        char_count=22,
        article_url="https://example.com/1",
        source="naver_news",
        title="평택을 판세 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보A",
        district_id="pyeongtaek_b",
        chunker_type="korean_paragraph",
    ),
]

SAMPLE_EMBEDDED = [
    ChunkWithEmbedding(
        text="평택을 선거구의 여론조사 결과가 발표되었다.",
        chunk_index=0,
        char_count=22,
        article_url="https://example.com/1",
        source="naver_news",
        title="평택을 판세 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보A",
        district_id="pyeongtaek_b",
        chunker_type="korean_paragraph",
        embedding=[0.1] * 1536,
    ),
]


class TestIngestionPipeline:

    def test_full_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1),
        ):
            pipeline.run()

        assert (tmp_path / f"articles_{pipeline._timestamp}.jsonl").exists()

    def test_skip_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk") as mock_chunk,
        ):
            pipeline.run(skip_chunk=True)

        mock_chunk.assert_not_called()
        assert (tmp_path / f"articles_{pipeline._timestamp}.jsonl").exists()
        assert not (tmp_path / f"chunks_{pipeline._timestamp}.jsonl").exists()

    def test_skip_embed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed") as mock_embed,
        ):
            pipeline.run(skip_embed=True)

        mock_embed.assert_not_called()
        assert (tmp_path / f"chunks_{pipeline._timestamp}.jsonl").exists()
        assert not (tmp_path / f"embeddings_{pipeline._timestamp}.jsonl").exists()

    def test_no_articles_stops_early(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=[]),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk") as mock_chunk,
        ):
            pipeline.run()

        mock_chunk.assert_not_called()

    def test_empty_chunks_stops_early(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=[]),
            patch.object(pipeline, "_embed") as mock_embed,
        ):
            pipeline.run()

        mock_embed.assert_not_called()

    def test_scraper_naver_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES) as mock_naver,
            patch.object(pipeline, "_run_political") as mock_political,
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1),
        ):
            pipeline.run(scraper_name="naver")

        mock_naver.assert_called_once()
        mock_political.assert_not_called()

    def test_scraper_political_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver") as mock_naver,
            patch.object(pipeline, "_run_political", return_value=SAMPLE_ARTICLES) as mock_political,
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1),
        ):
            pipeline.run(scraper_name="political")

        mock_naver.assert_not_called()
        mock_political.assert_called_once()

    def test_save_jsonl_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1),
        ):
            pipeline.run()

        articles_path = tmp_path / f"articles_{pipeline._timestamp}.jsonl"
        import json
        lines = articles_path.read_text("utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["url"] == "https://example.com/1"
        assert data["source"] == "naver_news"

    def test_no_keywords_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        empty_config = {**SAMPLE_CONFIG, "districts": []}
        pipeline = IngestionPipeline(empty_config)
        pipeline.run()

        assert not list(tmp_path.glob("*.jsonl"))

    def test_chunk_integration(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1),
        ):
            pipeline.run()

        chunks_path = tmp_path / f"chunks_{pipeline._timestamp}.jsonl"
        assert chunks_path.exists()
        import json
        lines = chunks_path.read_text("utf-8").strip().split("\n")
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["chunker_type"] == "korean_paragraph"

    def test_embed_integration(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_store", return_value=1),
            patch("openai.OpenAI") as MockClient,
        ):
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            MockClient.return_value = mock_client
            pipeline.run()

        embeddings_path = tmp_path / f"embeddings_{pipeline._timestamp}.jsonl"
        assert embeddings_path.exists()
        import json
        lines = embeddings_path.read_text("utf-8").strip().split("\n")
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert "embedding" in data
        assert len(data["embedding"]) == 1536

    def test_skip_store(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store") as mock_store,
        ):
            pipeline.run(skip_store=True)

        mock_store.assert_not_called()

    def test_store_called_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(SAMPLE_CONFIG)

        with (
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1) as mock_store,
        ):
            pipeline.run()

        mock_store.assert_called_once_with(SAMPLE_EMBEDDED)
