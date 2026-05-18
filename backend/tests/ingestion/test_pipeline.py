from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ingestion.pipeline import IngestionPipeline
from models.article import RawArticle
from models.chunk import Chunk, ChunkWithEmbedding
from models.poll import PollMeta


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

_SAMPLE_TEXT = "평택을 선거구의 여론조사 결과가 발표되었다. 후보A가 30%로 선두를 달리고 있으며 접전이 예상된다."

SAMPLE_CHUNKS = [
    Chunk(
        text=_SAMPLE_TEXT,
        chunk_index=0,
        char_count=len(_SAMPLE_TEXT),
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
        text=_SAMPLE_TEXT,
        chunk_index=0,
        char_count=len(_SAMPLE_TEXT),
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


class TestFilterChunks:

    def _make_chunk(self, text: str = "충분히 긴 텍스트입니다. " * 5, district_id: str = "pyeongtaek_b", **kwargs):
        defaults = {
            "chunk_index": 0,
            "char_count": len(text),
            "article_url": "https://example.com/1",
            "source": "naver_news",
            "title": "테스트",
            "published_at": datetime(2026, 5, 1),
            "candidate": "",
            "chunker_type": "korean_paragraph",
        }
        defaults.update(kwargs)
        return Chunk(text=text, district_id=district_id, **defaults)

    def test_keeps_valid_chunks(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        chunks = [self._make_chunk()]
        result = pipeline._filter_chunks(chunks)
        assert len(result) == 1

    def test_removes_short_chunks(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        short_text = "짧은 텍스트"
        chunks = [self._make_chunk(text=short_text)]
        result = pipeline._filter_chunks(chunks)
        assert len(result) == 0

    def test_removes_untagged_chunks(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        chunks = [self._make_chunk(district_id="")]
        result = pipeline._filter_chunks(chunks)
        assert len(result) == 0

    def test_mixed_filtering(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        chunks = [
            self._make_chunk(),
            self._make_chunk(text="짧음"),
            self._make_chunk(district_id=""),
            self._make_chunk(text="짧음", district_id=""),
            self._make_chunk(district_id="busan_bukgu_gap"),
        ]
        result = pipeline._filter_chunks(chunks)
        assert len(result) == 2

    def test_empty_input(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        assert pipeline._filter_chunks([]) == []

    def test_boundary_exactly_50_chars(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        text_50 = "가" * 50
        chunks = [self._make_chunk(text=text_50)]
        result = pipeline._filter_chunks(chunks)
        assert len(result) == 1

    def test_boundary_49_chars_removed(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        text_49 = "가" * 49
        chunks = [self._make_chunk(text=text_49)]
        result = pipeline._filter_chunks(chunks)
        assert len(result) == 0


# ── 여론조사 동기화 테스트 ────────────────────────────

SAMPLE_POLL_META = [
    PollMeta(
        survey_date=datetime(2026, 5, 14).date(),
        district_id="pyeongtaek_b",
        pollster="뉴스1",
        district_name="평택을",
        sample_size=804,
        margin_of_error=3.5,
        source_url="https://example.com/poll/1",
    ),
    PollMeta(
        survey_date=datetime(2026, 5, 7).date(),
        district_id="busan_bukgu_gap",
        pollster="SBS",
        district_name="부산북구갑",
        sample_size=802,
        margin_of_error=3.5,
        source_url="",
    ),
]

POLLS_CONFIG = {
    **SAMPLE_CONFIG,
    "polls": {
        "type": "google_sheets",
        "params": {
            "spreadsheet_id": "fake_id",
            "credentials_path": "fake.json",
        },
    },
}


class TestPollSync:

    def test_sync_polls_creates_articles_from_source_urls(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(POLLS_CONFIG)
        body = "여론조사 결과에 따르면 김용남 후보가 29%로 선두를 달리고 있다. " * 5

        with (
            patch("rag.gsheets_poll_store.GoogleSheetsPollStore") as MockGSheets,
            patch("rag.jsonl_poll_store.JsonlPollStore") as MockJsonl,
            patch("ingestion.scraper.base.fetch_article_body", return_value=body),
            patch("ingestion.scraper.url_store.ScrapedUrlStore") as MockUrlStore,
        ):
            mock_gs = MagicMock()
            mock_gs.load_all.return_value = []
            mock_gs.load_meta.return_value = SAMPLE_POLL_META
            MockGSheets.return_value = mock_gs

            mock_jsonl = MagicMock()
            MockJsonl.return_value = mock_jsonl

            mock_url_store = MagicMock()
            mock_url_store.load.return_value = set()
            MockUrlStore.return_value = mock_url_store

            articles = pipeline._sync_polls()

        assert len(articles) == 1
        assert articles[0].source == "poll"
        assert articles[0].pollster == "뉴스1"
        assert articles[0].poll_survey_date == "2026-05-14"
        assert articles[0].sample_size == 804
        assert articles[0].margin_of_error == 3.5
        assert articles[0].district_id == "pyeongtaek_b"
        mock_jsonl.save.assert_called_once()

    def test_sync_polls_skips_already_scraped_urls(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(POLLS_CONFIG)

        with (
            patch("rag.gsheets_poll_store.GoogleSheetsPollStore") as MockGSheets,
            patch("rag.jsonl_poll_store.JsonlPollStore") as MockJsonl,
            patch("ingestion.scraper.base.fetch_article_body") as mock_fetch,
            patch("ingestion.scraper.url_store.ScrapedUrlStore") as MockUrlStore,
        ):
            mock_gs = MagicMock()
            mock_gs.load_all.return_value = []
            mock_gs.load_meta.return_value = SAMPLE_POLL_META
            MockGSheets.return_value = mock_gs
            MockJsonl.return_value = MagicMock()

            mock_url_store = MagicMock()
            mock_url_store.load.return_value = {"https://example.com/poll/1"}
            MockUrlStore.return_value = mock_url_store

            articles = pipeline._sync_polls()

        assert len(articles) == 0
        mock_fetch.assert_not_called()

    def test_sync_polls_skips_empty_source_urls(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        metas_no_url = [
            PollMeta(
                survey_date=datetime(2026, 5, 14).date(),
                district_id="pyeongtaek_b",
                pollster="뉴스1",
                source_url="",
            ),
        ]

        pipeline = IngestionPipeline(POLLS_CONFIG)

        with (
            patch("rag.gsheets_poll_store.GoogleSheetsPollStore") as MockGSheets,
            patch("rag.jsonl_poll_store.JsonlPollStore") as MockJsonl,
            patch("ingestion.scraper.base.fetch_article_body") as mock_fetch,
            patch("ingestion.scraper.url_store.ScrapedUrlStore") as MockUrlStore,
        ):
            mock_gs = MagicMock()
            mock_gs.load_all.return_value = []
            mock_gs.load_meta.return_value = metas_no_url
            MockGSheets.return_value = mock_gs
            MockJsonl.return_value = MagicMock()
            mock_url_store = MagicMock()
            mock_url_store.load.return_value = set()
            MockUrlStore.return_value = mock_url_store

            articles = pipeline._sync_polls()

        assert len(articles) == 0
        mock_fetch.assert_not_called()

    def test_sync_polls_skipped_for_jsonl_type(self):
        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        articles = pipeline._sync_polls()
        assert articles == []

    def test_skip_polls_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(POLLS_CONFIG)

        with (
            patch.object(pipeline, "_sync_polls") as mock_sync,
            patch.object(pipeline, "_run_naver", return_value=SAMPLE_ARTICLES),
            patch.object(pipeline, "_run_political", return_value=[]),
            patch.object(pipeline, "_chunk", return_value=SAMPLE_CHUNKS),
            patch.object(pipeline, "_embed", return_value=SAMPLE_EMBEDDED),
            patch.object(pipeline, "_store", return_value=1),
        ):
            pipeline.run(skip_polls=True)

        mock_sync.assert_not_called()

    def test_poll_metadata_in_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        poll_article = RawArticle(
            url="https://example.com/poll/1",
            source="poll",
            title="[여론조사] 뉴스1 2026-05-14 평택을",
            body="평택을 여론조사 결과가 발표되었다. 김용남 후보가 29% 선두. " * 5,
            published_at=datetime(2026, 5, 14),
            district_id="pyeongtaek_b",
            pollster="뉴스1",
            poll_survey_date="2026-05-14",
            sample_size=804,
            margin_of_error=3.5,
        )

        pipeline = IngestionPipeline(SAMPLE_CONFIG)
        chunks = pipeline._chunk([poll_article])

        assert len(chunks) >= 1
        assert chunks[0].pollster == "뉴스1"
        assert chunks[0].poll_survey_date == "2026-05-14"
        assert chunks[0].sample_size == 804
        assert chunks[0].margin_of_error == 3.5
        assert chunks[0].source == "poll"

    def test_poll_article_body_too_short_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.pipeline.DATA_DIR", tmp_path)

        pipeline = IngestionPipeline(POLLS_CONFIG)

        with (
            patch("rag.gsheets_poll_store.GoogleSheetsPollStore") as MockGSheets,
            patch("rag.jsonl_poll_store.JsonlPollStore") as MockJsonl,
            patch("ingestion.scraper.base.fetch_article_body", return_value="짧은 본문"),
            patch("ingestion.scraper.url_store.ScrapedUrlStore") as MockUrlStore,
        ):
            mock_gs = MagicMock()
            mock_gs.load_all.return_value = []
            mock_gs.load_meta.return_value = SAMPLE_POLL_META
            MockGSheets.return_value = mock_gs
            MockJsonl.return_value = MagicMock()
            mock_url_store = MagicMock()
            mock_url_store.load.return_value = set()
            MockUrlStore.return_value = mock_url_store

            articles = pipeline._sync_polls()

        assert len(articles) == 0
