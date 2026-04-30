import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.base_registry import ComponentRegistry
from ingestion.scraper.base import AbstractScraper, ScraperRegistry
from ingestion.scraper.naver import NaverNewsScraper
from ingestion.scraper.political import PoliticalNewsScraper
from ingestion.scraper.url_store import ScrapedUrlStore
from models.article import RawArticle


# ============================================================
# Fixtures & Mock 데이터
# ============================================================

NAVER_MOCK_HTML = """
<div class="group_news">
<ul class="list_news _infinite_list">
<div class="_slog_visible">
<div class="sds-comps-vertical-layout">
  <a data-heatmap-target=".prof" href="https://media.naver.com/press/001">
    <span class="sds-comps-text sds-comps-profile-info-title-text">테스트언론</span>
    <span class="sds-comps-text sds-comps-profile-info-subtext">2026.05.01.</span>
  </a>
  <a data-heatmap-target=".tit" href="https://example.com/article/1">평택을 재보궐 후보 확정</a>
  <a data-heatmap-target=".body" href="https://example.com/article/1">평택을 재보궐선거 후보가 확정되었습니다.</a>
</div>
<div class="sds-comps-vertical-layout">
  <a data-heatmap-target=".prof" href="https://media.naver.com/press/002">
    <span class="sds-comps-text sds-comps-profile-info-title-text">뉴스매체</span>
    <span class="sds-comps-text sds-comps-profile-info-subtext">2026.05.02.</span>
  </a>
  <a data-heatmap-target=".tit" href="https://example.com/article/2">부산 북구갑 선거 소식</a>
  <a data-heatmap-target=".body" href="https://example.com/article/2">부산 북구갑 재보궐 관련 소식입니다.</a>
</div>
</div>
</ul>
</div>
"""

NAVER_EMPTY_HTML = """<div class="group_news"><ul class="list_news _infinite_list"></ul></div>"""

RSS_MOCK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>오마이뉴스</title>
    <item>
      <title>평택을 후보 공약 발표</title>
      <link>https://www.ohmynews.com/article/1</link>
      <description>평택을 재보궐 후보가 공약을 발표했습니다.</description>
      <pubDate>Fri, 01 May 2026 09:00:00 +0900</pubDate>
    </item>
    <item>
      <title>경제 뉴스 제목</title>
      <link>https://www.ohmynews.com/article/2</link>
      <description>경제 관련 기사입니다.</description>
      <pubDate>Fri, 01 May 2026 10:00:00 +0900</pubDate>
    </item>
    <item>
      <title>부산 북구갑 선거 동향</title>
      <link>https://www.ohmynews.com/article/3</link>
      <description>부산 북구갑 재보궐 동향을 살펴봅니다.</description>
      <pubDate>Sat, 02 May 2026 11:00:00 +0900</pubDate>
    </item>
  </channel>
</rss>
"""


def _mock_httpx_get_naver(url, **kwargs):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = NAVER_MOCK_HTML
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _mock_httpx_get_naver_empty(url, **kwargs):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = NAVER_EMPTY_HTML
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _mock_httpx_get_rss(url, **kwargs):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = RSS_MOCK_XML
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ============================================================
# ComponentRegistry 테스트
# ============================================================


class TestComponentRegistry:
    def test_register_and_create(self):
        registry = ComponentRegistry(AbstractScraper, "TestScraper")

        @registry.register("dummy")
        class DummyScraper(AbstractScraper):
            def __init__(self, **kwargs):
                pass

            def scrape(self, keywords, date_from=None, date_to=None, max_articles=50):
                return []

            @property
            def source_name(self):
                return "dummy"

        instance = registry.create("dummy")
        assert isinstance(instance, DummyScraper)
        assert instance.source_name == "dummy"

    def test_create_unknown_raises(self):
        registry = ComponentRegistry(AbstractScraper, "TestScraper")
        with pytest.raises(ValueError, match="미등록"):
            registry.create("nonexistent")

    def test_registered_names(self):
        assert "naver" in ScraperRegistry.registered_names
        assert "political" in ScraperRegistry.registered_names


# ============================================================
# ScrapedUrlStore 테스트
# ============================================================


class TestScrapedUrlStore:
    def test_add_and_contains(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        store = ScrapedUrlStore(store_path)

        assert not store.contains("https://example.com/1")
        store.add("https://example.com/1", "test", "제목1")
        assert store.contains("https://example.com/1")
        assert store.count == 1

    def test_add_batch(self, tmp_path: Path):
        store = ScrapedUrlStore(tmp_path / "urls.jsonl")

        articles = [
            {"url": "https://a.com/1", "source": "s1", "title": "t1"},
            {"url": "https://a.com/2", "source": "s1", "title": "t2"},
            {"url": "https://a.com/1", "source": "s1", "title": "t1"},
        ]
        saved = store.add_batch(articles)

        assert saved == 2
        assert store.count == 2

    def test_persistence(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        store1 = ScrapedUrlStore(store_path)
        store1.add("https://example.com/1", "test", "제목")

        store2 = ScrapedUrlStore(store_path)
        assert store2.contains("https://example.com/1")
        assert store2.count == 1

    def test_jsonl_format(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        store = ScrapedUrlStore(store_path)
        store.add("https://example.com/1", "naver", "테스트 기사")

        lines = store_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["url"] == "https://example.com/1"
        assert record["source"] == "naver"
        assert record["title"] == "테스트 기사"
        assert "scraped_at" in record

    def test_empty_store_file_not_exist(self, tmp_path: Path):
        store = ScrapedUrlStore(tmp_path / "nonexistent.jsonl")
        assert store.count == 0
        assert not store.contains("https://example.com")


# ============================================================
# NaverNewsScraper 테스트
# ============================================================


class TestNaverNewsScraper:
    def test_scrape_returns_articles(self, tmp_path: Path):
        scraper = NaverNewsScraper(max_articles_per_run=10, request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=_mock_httpx_get_naver):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert len(results) == 2
        assert results[0].title == "평택을 재보궐 후보 확정"
        assert results[0].source == "naver_news"
        assert results[0].url == "https://example.com/article/1"
        assert results[0].matched_keywords == ["평택을"]

    def test_scrape_respects_max_articles(self, tmp_path: Path):
        scraper = NaverNewsScraper(max_articles_per_run=1, request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=_mock_httpx_get_naver):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert len(results) <= 1

    def test_scrape_returns_empty_on_http_error(self, tmp_path: Path):
        scraper = NaverNewsScraper(request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=Exception("Connection error")):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert results == []

    def test_scrape_stops_on_429(self, tmp_path: Path):
        def mock_429(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            return mock_resp

        scraper = NaverNewsScraper(request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=mock_429):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 1),
            )

        assert results == []

    def test_scrape_deduplicates_urls(self, tmp_path: Path):
        scraper = NaverNewsScraper(max_articles_per_run=50, request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=_mock_httpx_get_naver):
            results = scraper.scrape(
                keywords=["평택을", "재보궐"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    def test_scrape_stops_on_empty_page(self, tmp_path: Path):
        scraper = NaverNewsScraper(max_articles_per_run=50, request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=_mock_httpx_get_naver_empty):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 1),
            )

        assert results == []

    def test_parse_date_from_html(self, tmp_path: Path):
        scraper = NaverNewsScraper(max_articles_per_run=10, request_delay_sec=0, url_store_path=tmp_path / "urls.jsonl")

        with patch("httpx.get", side_effect=_mock_httpx_get_naver):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert results[0].published_at == datetime(2026, 5, 1)
        assert results[1].published_at == datetime(2026, 5, 2)

    def test_scrape_saves_urls_to_store(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        scraper = NaverNewsScraper(max_articles_per_run=10, request_delay_sec=0, url_store_path=store_path)

        with patch("httpx.get", side_effect=_mock_httpx_get_naver):
            scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        store = ScrapedUrlStore(store_path)
        assert store.contains("https://example.com/article/1")
        assert store.contains("https://example.com/article/2")

    def test_scrape_skips_already_stored_urls(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        store = ScrapedUrlStore(store_path)
        store.add("https://example.com/article/1", "naver_news", "이전 기사")

        scraper = NaverNewsScraper(max_articles_per_run=10, request_delay_sec=0, url_store_path=store_path)

        with patch("httpx.get", side_effect=_mock_httpx_get_naver):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert len(results) == 1
        assert results[0].url == "https://example.com/article/2"

    def test_scrape_default_date_range(self, tmp_path: Path):
        scraper = NaverNewsScraper(
            request_delay_sec=0,
            lookback_days=2,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_naver_empty):
            scraper.scrape(keywords=["평택을"])

        resolved_from, resolved_to = scraper.resolve_date_range(None, None)
        assert resolved_to == date.today()
        assert resolved_from == date.today() - timedelta(days=2)


# ============================================================
# PoliticalNewsScraper 테스트
# ============================================================


class TestPoliticalNewsScraper:
    def test_scrape_returns_matching_articles(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            max_articles_per_run=30,
            request_delay_sec=0,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            results = scraper.scrape(
                keywords=["평택을", "부산 북구갑"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert len(results) == 2
        titles = {r.title for r in results}
        assert "평택을 후보 공약 발표" in titles
        assert "부산 북구갑 선거 동향" in titles

    def test_scrape_filters_by_keyword(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            request_delay_sec=0,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            results = scraper.scrape(
                keywords=["경제"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert len(results) == 1
        assert results[0].title == "경제 뉴스 제목"

    def test_scrape_filters_by_date_range(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            request_delay_sec=0,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            results = scraper.scrape(
                keywords=["평택을", "부산 북구갑"],
                date_from=date(2026, 5, 2),
                date_to=date(2026, 5, 2),
            )

        assert len(results) == 1
        assert results[0].title == "부산 북구갑 선거 동향"

    def test_scrape_returns_empty_on_error(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            request_delay_sec=0,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=Exception("timeout")):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 1),
            )

        assert results == []

    def test_scrape_multiple_sources(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com", "https://www.pressian.com"],
            request_delay_sec=0,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            results = scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        sources = {r.source for r in results}
        assert "ohmynews" in sources
        assert "pressian" in sources

    def test_extract_source_label(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(url_store_path=tmp_path / "urls.jsonl")
        assert scraper._extract_source_label("https://www.ohmynews.com") == "ohmynews"
        assert scraper._extract_source_label("https://www.pressian.com/path") == "pressian"
        assert scraper._extract_source_label("https://www.mediatoday.co.kr") == "mediatoday"

    def test_scrape_respects_max_articles(self, tmp_path: Path):
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            max_articles_per_run=1,
            request_delay_sec=0,
            url_store_path=tmp_path / "urls.jsonl",
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            results = scraper.scrape(
                keywords=["평택을", "부산 북구갑", "경제"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        assert len(results) <= 1

    def test_scrape_saves_urls_to_store(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            request_delay_sec=0,
            url_store_path=store_path,
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            scraper.scrape(
                keywords=["평택을"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        store = ScrapedUrlStore(store_path)
        assert store.contains("https://www.ohmynews.com/article/1")

    def test_scrape_skips_already_stored_urls(self, tmp_path: Path):
        store_path = tmp_path / "urls.jsonl"
        store = ScrapedUrlStore(store_path)
        store.add("https://www.ohmynews.com/article/1", "ohmynews", "이전 기사")

        scraper = PoliticalNewsScraper(
            urls=["https://www.ohmynews.com"],
            request_delay_sec=0,
            url_store_path=store_path,
        )

        with patch("httpx.get", side_effect=_mock_httpx_get_rss):
            results = scraper.scrape(
                keywords=["평택을", "부산 북구갑"],
                date_from=date(2026, 5, 1),
                date_to=date(2026, 5, 2),
            )

        urls = [r.url for r in results]
        assert "https://www.ohmynews.com/article/1" not in urls


# ============================================================
# lookback_days / resolve_date_range 테스트
# ============================================================


class TestDateRange:
    def test_resolve_with_explicit_dates(self, tmp_path: Path):
        scraper = NaverNewsScraper(request_delay_sec=0, url_store_path=tmp_path / "u.jsonl")
        d_from, d_to = scraper.resolve_date_range(date(2026, 5, 1), date(2026, 5, 3))
        assert d_from == date(2026, 5, 1)
        assert d_to == date(2026, 5, 3)

    def test_resolve_with_none_uses_lookback(self, tmp_path: Path):
        scraper = NaverNewsScraper(request_delay_sec=0, lookback_days=3, url_store_path=tmp_path / "u.jsonl")
        d_from, d_to = scraper.resolve_date_range(None, None)
        assert d_to == date.today()
        assert d_from == date.today() - timedelta(days=3)

    def test_resolve_partial_none(self, tmp_path: Path):
        scraper = NaverNewsScraper(request_delay_sec=0, url_store_path=tmp_path / "u.jsonl")
        d_from, d_to = scraper.resolve_date_range(date(2026, 4, 1), None)
        assert d_from == date(2026, 4, 1)
        assert d_to == date.today()

    def test_default_lookback_is_two_days(self, tmp_path: Path):
        scraper = NaverNewsScraper(request_delay_sec=0, url_store_path=tmp_path / "u.jsonl")
        assert scraper.lookback_days == 2


# ============================================================
# RawArticle 모델 테스트
# ============================================================


class TestRawArticle:
    def test_default_values(self):
        article = RawArticle(
            url="https://example.com/1",
            source="test",
            title="제목",
            body="본문",
            published_at=datetime(2026, 5, 1),
        )
        assert article.candidate == ""
        assert article.district_id == ""
        assert article.matched_keywords == []

    def test_full_construction(self):
        article = RawArticle(
            url="https://example.com/1",
            source="naver_news",
            title="평택을 기사",
            body="본문 내용",
            published_at=datetime(2026, 5, 1),
            candidate="홍길동",
            district_id="pyeongtaek_b",
            matched_keywords=["평택을", "홍길동"],
        )
        assert article.candidate == "홍길동"
        assert article.district_id == "pyeongtaek_b"
        assert len(article.matched_keywords) == 2
