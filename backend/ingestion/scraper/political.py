from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingestion.scraper.base import AbstractScraper, ScraperRegistry
from ingestion.scraper.url_store import ScrapedUrlStore
from models.article import RawArticle

logger = logging.getLogger(__name__)


@ScraperRegistry.register("political")
class PoliticalNewsScraper(AbstractScraper):

    def __init__(
        self,
        urls: list[str] | None = None,
        max_articles_per_run: int = 30,
        request_delay_sec: float = 1.5,
        lookback_days: int = 2,
        url_store_path: str | Path | None = None,
    ) -> None:
        self._urls = urls or []
        self._max = max_articles_per_run
        self._delay = request_delay_sec
        self._lookback_days = lookback_days
        self._url_store = ScrapedUrlStore(url_store_path)
        logger.info(
            "[%s] 초기화 완료 — 매체 %d곳, max=%d, lookback=%d일, 기존 URL=%d건",
            self.source_name, len(self._urls), self._max, self._lookback_days, self._url_store.count,
        )

    @property
    def source_name(self) -> str:
        return "political_news"

    @property
    def request_delay_sec(self) -> float:
        return self._delay

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    def scrape(
        self,
        keywords: list[str],
        date_from: date | None = None,
        date_to: date | None = None,
        max_articles: int = 50,
    ) -> list[RawArticle]:
        resolved_from, resolved_to = self.resolve_date_range(date_from, date_to)
        limit = min(max_articles, self._max)

        logger.info(
            "[%s] 수집 시작 — keywords=%s, 매체=%d곳, 기간=%s~%s, limit=%d",
            self.source_name, keywords, len(self._urls), resolved_from, resolved_to, limit,
        )

        articles: list[RawArticle] = []
        skipped_existing = 0

        for base_url in self._urls:
            if len(articles) >= limit:
                logger.info("[%s] limit 도달 — 매체 루프 중단", self.source_name)
                break
            try:
                source_label = self._extract_source_label(base_url)
                logger.info("[%s] RSS 수집: %s (%s)", self.source_name, source_label, base_url)

                fetched = self._fetch_from_rss(base_url, keywords, resolved_from, resolved_to)

                for article in fetched:
                    if self._url_store.contains(article.url):
                        skipped_existing += 1
                        continue
                    articles.append(article)

                logger.info(
                    "[%s] %s — %d건 수집, 누적 %d건",
                    self.source_name, source_label, len(fetched), len(articles),
                )
                time.sleep(self.request_delay_sec)
            except Exception as e:
                logger.warning("[%s] %s 수집 실패: %s", self.source_name, base_url, e)

        result = articles[:limit]

        saved = self._url_store.add_batch([
            {"url": a.url, "source": a.source, "title": a.title} for a in result
        ])

        logger.info(
            "[%s] 수집 완료 — 총 %d건 반환, 신규 URL %d건 저장, 기존 URL %d건 스킵",
            self.source_name, len(result), saved, skipped_existing,
        )
        return result

    def _fetch_from_rss(
        self,
        base_url: str,
        keywords: list[str],
        date_from: date,
        date_to: date,
    ) -> list[RawArticle]:
        import feedparser
        import httpx

        rss_url = f"{base_url.rstrip('/')}/rss"
        logger.debug("[%s] RSS 요청: %s", self.source_name, rss_url)
        resp = httpx.get(rss_url, timeout=10)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        logger.debug("[%s] RSS 엔트리 %d건 수신", self.source_name, len(feed.entries))

        articles: list[RawArticle] = []
        source_label = self._extract_source_label(base_url)

        for entry in feed.entries:
            title: str = entry.get("title", "")
            link: str = entry.get("link", "")
            summary: str = entry.get("summary", "")

            matched = [kw for kw in keywords if kw in title or kw in summary]
            if not matched:
                continue

            published_at = self._parse_entry_date(entry)
            if published_at.date() < date_from or published_at.date() > date_to:
                logger.debug("[%s] 날짜 범위 밖 스킵: %s (%s)", self.source_name, title[:30], published_at.date())
                continue

            articles.append(
                RawArticle(
                    url=link,
                    source=source_label,
                    title=title,
                    body=summary,
                    published_at=published_at,
                    matched_keywords=matched,
                )
            )

        return articles

    def _parse_entry_date(self, entry: Any) -> datetime:
        from email.utils import parsedate_to_datetime

        published = entry.get("published", "")
        if published:
            try:
                return parsedate_to_datetime(published)
            except (ValueError, TypeError):
                logger.debug("[%s] published 날짜 파싱 실패: '%s'", self.source_name, published)

        updated = entry.get("updated", "")
        if updated:
            try:
                return parsedate_to_datetime(updated)
            except (ValueError, TypeError):
                logger.debug("[%s] updated 날짜 파싱 실패: '%s'", self.source_name, updated)

        logger.debug("[%s] 날짜 정보 없음 — 현재 시각으로 대체", self.source_name)
        return datetime.now()

    def _extract_source_label(self, url: str) -> str:
        from urllib.parse import urlparse

        hostname = urlparse(url).hostname or ""
        parts = hostname.replace("www.", "").split(".")
        return parts[0] if parts else "unknown"
