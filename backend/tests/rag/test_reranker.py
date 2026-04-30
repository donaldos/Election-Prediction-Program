from __future__ import annotations

from datetime import datetime

import pytest

from models.score import SearchResult
from rag.reranker import Reranker


def _make_result(
    id: str = "id-1",
    score: float = 0.8,
    article_url: str = "https://example.com/1",
    candidate: str = "후보A",
    **kwargs,
) -> SearchResult:
    defaults = dict(
        text="테스트 청크 텍스트",
        source="naver_news",
        title="테스트 기사",
        published_at=datetime(2026, 4, 28),
        district_id="pyeongtaek_b",
    )
    defaults.update(kwargs)
    return SearchResult(
        id=id,
        score=score,
        article_url=article_url,
        candidate=candidate,
        **defaults,
    )


class TestRerankerScoreFiltering:

    def test_filters_below_threshold(self):
        results = [
            _make_result(id="1", score=0.9),
            _make_result(id="2", score=0.1),
            _make_result(id="3", score=0.5),
        ]
        reranker = Reranker(min_score=0.3, deduplicate=False)
        filtered = reranker.rerank("질의", results)

        assert len(filtered) == 2
        assert all(r.score >= 0.3 for r in filtered)

    def test_all_below_threshold_returns_empty(self):
        results = [
            _make_result(id="1", score=0.1),
            _make_result(id="2", score=0.2),
        ]
        reranker = Reranker(min_score=0.5, deduplicate=False)
        filtered = reranker.rerank("질의", results)

        assert filtered == []

    def test_empty_input_returns_empty(self):
        reranker = Reranker()
        assert reranker.rerank("질의", []) == []

    def test_zero_threshold_keeps_all(self):
        results = [
            _make_result(id="1", score=0.01),
            _make_result(id="2", score=0.99),
        ]
        reranker = Reranker(min_score=0.0, deduplicate=False)
        filtered = reranker.rerank("질의", results)

        assert len(filtered) == 2


class TestRerankerDeduplication:

    def test_keeps_highest_score_per_url(self):
        results = [
            _make_result(id="1", score=0.6, article_url="https://a.com/1"),
            _make_result(id="2", score=0.9, article_url="https://a.com/1"),
            _make_result(id="3", score=0.7, article_url="https://b.com/2"),
        ]
        reranker = Reranker(min_score=0.0, deduplicate=True)
        filtered = reranker.rerank("질의", results)

        assert len(filtered) == 2
        urls = {r.article_url for r in filtered}
        assert urls == {"https://a.com/1", "https://b.com/2"}

        a_result = next(r for r in filtered if r.article_url == "https://a.com/1")
        assert a_result.id == "2"
        assert a_result.score == 0.9

    def test_no_dedup_when_disabled(self):
        results = [
            _make_result(id="1", score=0.6, article_url="https://a.com/1"),
            _make_result(id="2", score=0.9, article_url="https://a.com/1"),
        ]
        reranker = Reranker(min_score=0.0, deduplicate=False)
        filtered = reranker.rerank("질의", results)

        assert len(filtered) == 2

    def test_unique_urls_unchanged(self):
        results = [
            _make_result(id="1", score=0.8, article_url="https://a.com/1"),
            _make_result(id="2", score=0.7, article_url="https://b.com/2"),
            _make_result(id="3", score=0.6, article_url="https://c.com/3"),
        ]
        reranker = Reranker(min_score=0.0, deduplicate=True)
        filtered = reranker.rerank("질의", results)

        assert len(filtered) == 3


class TestRerankerSorting:

    def test_sorted_by_score_descending(self):
        results = [
            _make_result(id="1", score=0.5),
            _make_result(id="2", score=0.9),
            _make_result(id="3", score=0.7),
        ]
        reranker = Reranker(min_score=0.0, deduplicate=False)
        filtered = reranker.rerank("질의", results)

        scores = [r.score for r in filtered]
        assert scores == [0.9, 0.7, 0.5]


class TestRerankerIntegrated:

    def test_filter_dedup_sort_combined(self):
        """임계값 필터링 + 중복 제거 + 정렬이 함께 동작."""
        results = [
            _make_result(id="1", score=0.1, article_url="https://a.com/1"),
            _make_result(id="2", score=0.8, article_url="https://a.com/1"),
            _make_result(id="3", score=0.6, article_url="https://b.com/2"),
            _make_result(id="4", score=0.2, article_url="https://c.com/3"),
            _make_result(id="5", score=0.9, article_url="https://d.com/4"),
        ]
        reranker = Reranker(min_score=0.3, deduplicate=True)
        filtered = reranker.rerank("질의", results)

        assert len(filtered) == 3
        assert [r.id for r in filtered] == ["5", "2", "3"]
