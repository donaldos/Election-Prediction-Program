from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models.score import SearchResult
from rag.retriever import Retriever


MOCK_VECTOR = [0.1] * 1536

SAMPLE_RAW_RESULTS = [
    {
        "id": "chunk-001",
        "score": 0.85,
        "text": "김용남 후보가 평택을에서 지지율 선두를 유지하고 있다.",
        "article_url": "https://example.com/news/1",
        "source": "naver_news",
        "title": "평택을 여론조사 결과",
        "published_at": "2026-04-28T09:00:00",
        "candidate": "김용남",
        "district_id": "pyeongtaek_b",
    },
    {
        "id": "chunk-002",
        "score": 0.72,
        "text": "유의동 후보가 평택을에서 추격하고 있다.",
        "article_url": "https://example.com/news/2",
        "source": "naver_news",
        "title": "평택을 접전 양상",
        "published_at": "2026-04-27T14:00:00",
        "candidate": "유의동",
        "district_id": "pyeongtaek_b",
    },
]


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = MOCK_VECTOR
    return embedder


def _make_repo(raw_results: list[dict] | None = None) -> MagicMock:
    repo = MagicMock()
    repo.search.return_value = raw_results if raw_results is not None else SAMPLE_RAW_RESULTS
    return repo


class TestRetrieverRetrieve:

    def test_calls_embed_query(self):
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo, top_k=10)

        retriever.retrieve("평택을 판세", district_id="pyeongtaek_b")

        embedder.embed_query.assert_called_once_with("평택을 판세")

    def test_calls_search_with_filters(self):
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo, top_k=15)

        retriever.retrieve("판세", district_id="pyeongtaek_b", candidate="김용남")

        repo.search.assert_called_once_with(
            query_vector=MOCK_VECTOR,
            top_k=15,
            filters={"district_id": "pyeongtaek_b", "candidate": "김용남"},
        )

    def test_search_without_candidate_filter(self):
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo, top_k=20)

        retriever.retrieve("판세", district_id="pyeongtaek_b")

        repo.search.assert_called_once_with(
            query_vector=MOCK_VECTOR,
            top_k=20,
            filters={"district_id": "pyeongtaek_b"},
        )

    def test_converts_dict_to_search_result(self):
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo)

        results = retriever.retrieve("판세", district_id="pyeongtaek_b")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

        first = results[0]
        assert first.id == "chunk-001"
        assert first.score == 0.85
        assert first.candidate == "김용남"
        assert first.district_id == "pyeongtaek_b"

    def test_empty_search_returns_empty_list(self):
        embedder = _make_embedder()
        repo = _make_repo(raw_results=[])
        retriever = Retriever(embedder=embedder, vector_repo=repo)

        results = retriever.retrieve("존재하지 않는 질의", district_id="pyeongtaek_b")

        assert results == []

    def test_skips_invalid_result(self):
        """변환 실패한 결과는 스킵하고 나머지를 반환."""
        raw = [
            {"id": "good", "score": 0.9, "text": "유효",
             "article_url": "https://a.com", "source": "s", "title": "t",
             "published_at": "2026-04-28T00:00:00",
             "candidate": "c", "district_id": "d"},
            {"id": "bad", "score": 0.8},
        ]
        embedder = _make_embedder()
        repo = _make_repo(raw_results=raw)
        retriever = Retriever(embedder=embedder, vector_repo=repo)

        results = retriever.retrieve("질의", district_id="d")

        assert len(results) == 1
        assert results[0].id == "good"

    def test_top_k_passed_through(self):
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo, top_k=5)

        retriever.retrieve("질의", district_id="pyeongtaek_b")

        _, kwargs = repo.search.call_args
        assert kwargs["top_k"] == 5


class TestRetrieverRetrieveForDistrict:

    DISTRICT = {
        "id": "pyeongtaek_b",
        "name": "평택을",
        "candidates": [
            {"name": "김용남", "party": "더불어민주당"},
            {"name": "유의동", "party": "국민의힘"},
        ],
    }

    def test_searches_per_candidate(self):
        embedder = _make_embedder()
        repo = _make_repo(raw_results=[])
        retriever = Retriever(embedder=embedder, vector_repo=repo, top_k=10)

        retriever.retrieve_for_district(self.DISTRICT)

        assert embedder.embed_query.call_count == 2
        # 필터 검색 0건 → fallback 재검색으로 후보당 2회씩 호출
        assert repo.search.call_count == 4

    def test_deduplicates_by_id(self):
        same_result = {
            "id": "shared-chunk",
            "score": 0.85,
            "text": "공통 기사",
            "article_url": "https://a.com/1",
            "source": "naver_news",
            "title": "공통",
            "published_at": "2026-04-28T00:00:00",
            "candidate": "김용남",
            "district_id": "pyeongtaek_b",
        }
        embedder = _make_embedder()
        repo = _make_repo(raw_results=[same_result])
        retriever = Retriever(embedder=embedder, vector_repo=repo)

        results = retriever.retrieve_for_district(self.DISTRICT)

        assert len(results) == 1

    def test_empty_candidates_returns_empty(self):
        district = {"id": "test", "name": "테스트구", "candidates": []}
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo)

        results = retriever.retrieve_for_district(district)

        assert results == []


class TestEmbedQuery:
    """AbstractEmbedder.embed_query 메서드 테스트."""

    def test_embed_query_calls_do_embed(self):
        from ingestion.embedder.base import AbstractEmbedder

        class FakeEmbedder(AbstractEmbedder):
            def __init__(self):
                self._loaded = True
                self._called_with = None

            @property
            def name(self) -> str:
                return "fake"

            def load(self) -> None:
                self._loaded = True

            @property
            def is_loaded(self) -> bool:
                return self._loaded

            @property
            def dimensions(self) -> int:
                return 4

            def _do_embed(self, texts: list[str]) -> list[list[float]]:
                self._called_with = texts
                return [[0.1, 0.2, 0.3, 0.4]]

        embedder = FakeEmbedder()
        vector = embedder.embed_query("테스트 질의")

        assert vector == [0.1, 0.2, 0.3, 0.4]
        assert embedder._called_with == ["테스트 질의"]

    def test_embed_query_triggers_load(self):
        from ingestion.embedder.base import AbstractEmbedder

        class LazyEmbedder(AbstractEmbedder):
            def __init__(self):
                self._loaded = False

            @property
            def name(self) -> str:
                return "lazy"

            def load(self) -> None:
                self._loaded = True

            @property
            def is_loaded(self) -> bool:
                return self._loaded

            @property
            def dimensions(self) -> int:
                return 2

            def _do_embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 2.0]]

        embedder = LazyEmbedder()
        assert not embedder.is_loaded

        embedder.embed_query("질의")

        assert embedder.is_loaded


class TestRetrieverLookbackDays:

    @patch("rag.retriever.date")
    def test_filters_old_articles(self, mock_date):
        mock_date.today.return_value = date(2026, 4, 30)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        old_result = {
            "id": "old", "score": 0.9, "text": "오래된 기사",
            "article_url": "https://a.com/old", "source": "s", "title": "t",
            "published_at": "2026-04-10T00:00:00",
            "candidate": "c", "district_id": "d",
        }
        recent_result = {
            "id": "new", "score": 0.8, "text": "최신 기사",
            "article_url": "https://a.com/new", "source": "s", "title": "t",
            "published_at": "2026-04-25T00:00:00",
            "candidate": "c", "district_id": "d",
        }
        embedder = _make_embedder()
        repo = _make_repo(raw_results=[old_result, recent_result])
        retriever = Retriever(embedder=embedder, vector_repo=repo, lookback_days=7)

        results = retriever.retrieve("질의", district_id="d")

        assert len(results) == 1
        assert results[0].id == "new"

    def test_no_filter_when_lookback_none(self):
        embedder = _make_embedder()
        repo = _make_repo()
        retriever = Retriever(embedder=embedder, vector_repo=repo, lookback_days=None)

        results = retriever.retrieve("판세", district_id="pyeongtaek_b")

        assert len(results) == 2

    @patch("rag.retriever.date")
    def test_all_filtered_returns_empty(self, mock_date):
        mock_date.today.return_value = date(2026, 4, 30)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        old_result = {
            "id": "old", "score": 0.9, "text": "오래된 기사",
            "article_url": "https://a.com/old", "source": "s", "title": "t",
            "published_at": "2026-03-01T00:00:00",
            "candidate": "c", "district_id": "d",
        }
        embedder = _make_embedder()
        repo = _make_repo(raw_results=[old_result])
        retriever = Retriever(embedder=embedder, vector_repo=repo, lookback_days=7)

        results = retriever.retrieve("질의", district_id="d")

        assert results == []
