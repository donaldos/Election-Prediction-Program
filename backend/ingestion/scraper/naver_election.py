from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from ingestion.scraper.base import AbstractScraper, ScraperRegistry, fetch_article_body
from ingestion.scraper.url_store import ScrapedUrlStore
from models.article import RawArticle

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://news.naver.com/election/region2026",
}

BASE_URL = "https://news.naver.com/election/region2026/news"

SEL_ARTICLE_ITEM = "li.news_article_item"
SEL_ARTICLE_LINK = "a.link_article_area"
SEL_TITLE = "span.article_title"
SEL_PRESS = "span.press"
SEL_TIME = "span.time"


@ScraperRegistry.register("naver_election")
class NaverElectionScraper(AbstractScraper):

    def __init__(
        self,
        max_articles_per_run: int = 100,
        request_delay_sec: float = 1.5,
        lookback_days: int = 7,
        url_store_path: str | Path | None = None,
    ) -> None:
        self._max = max_articles_per_run
        self._delay = request_delay_sec
        self._lookback_days = lookback_days
        self._url_store = ScrapedUrlStore(url_store_path)
        logger.info(
            "[%s] 초기화 완료 — max=%d, delay=%.1fs, lookback=%d일, 기존 URL=%d건",
            self.source_name, self._max, self._delay, self._lookback_days, self._url_store.count,
        )

    @property
    def source_name(self) -> str:
        return "naver_election"

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
        max_articles: int = 100,
    ) -> list[RawArticle]:
        resolved_from, resolved_to = self.resolve_date_range(date_from, date_to)
        limit = min(max_articles, self._max)

        logger.info(
            "[%s] 수집 시작 — 기간=%s~%s, limit=%d",
            self.source_name, resolved_from, resolved_to, limit,
        )

        articles = self._fetch_list_page(limit)

        cutoff = datetime.combine(resolved_from, datetime.min.time())
        articles = [a for a in articles if a.published_at >= cutoff]

        result: list[RawArticle] = []
        skipped_existing = 0
        for a in articles:
            if self._url_store.contains(a.url):
                skipped_existing += 1
                continue
            result.append(a)
            if len(result) >= limit:
                break

        self._enrich_bodies(result)

        saved = self._url_store.add_batch([
            {"url": a.url, "source": a.source, "title": a.title} for a in result
        ])

        logger.info(
            "[%s] 수집 완료 — 총 %d건 반환, 신규 URL %d건 저장, 기존 URL %d건 스킵",
            self.source_name, len(result), saved, skipped_existing,
        )
        return result

    def _enrich_bodies(self, articles: list[RawArticle]) -> None:
        enriched = 0
        for article in articles:
            body = fetch_article_body(article.url, headers=HEADERS, timeout=10)
            if body and len(body) > len(article.body):
                article.body = body
                enriched += 1
            time.sleep(self._delay)
        logger.info(
            "[%s] 기사 전문 수집 — %d/%d건 본문 확보",
            self.source_name, enriched, len(articles),
        )

    def _fetch_list_page(self, limit: int) -> list[RawArticle]:
        import httpx
        from bs4 import BeautifulSoup

        articles: list[RawArticle] = []

        try:
            logger.debug("[%s] HTTP 요청 — %s", self.source_name, BASE_URL)
            resp = httpx.get(BASE_URL, headers=HEADERS, timeout=10, follow_redirects=True)

            if resp.status_code in (404, 429):
                logger.warning("[%s] HTTP %d 응답", self.source_name, resp.status_code)
                return []
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select(SEL_ARTICLE_ITEM)

            if not items:
                logger.warning("[%s] 기사 항목 0건 — 셀렉터 변경 가능성 확인 필요", self.source_name)
                return []

            logger.info("[%s] 파싱 대상: %d건", self.source_name, len(items))

            for item in items:
                link_tag = item.select_one(SEL_ARTICLE_LINK)
                if not link_tag:
                    continue

                url = link_tag.get("href", "")
                if not url.startswith("http"):
                    url = "https://n.news.naver.com" + url

                title_tag = item.select_one(SEL_TITLE)
                title = title_tag.get_text(strip=True) if title_tag else ""

                press_tag = item.select_one(SEL_PRESS)
                press = press_tag.get_text(strip=True) if press_tag else ""

                published_at = self._parse_time(item)

                articles.append(RawArticle(
                    url=url,
                    source=self.source_name,
                    title=title,
                    body=f"[{press}] {title}",
                    published_at=published_at,
                    matched_keywords=[],
                ))

                if len(articles) >= limit:
                    break

        except Exception as e:
            logger.warning("[%s] 수집 중 오류: %s", self.source_name, e)

        return articles

    def _parse_time(self, item) -> datetime:
        now = datetime.now()
        time_tag = item.select_one(SEL_TIME)
        if not time_tag:
            return now

        text = time_tag.get_text(strip=True)

        match = re.search(r"(\d+)분\s*전", text)
        if match:
            return now - timedelta(minutes=int(match.group(1)))

        match = re.search(r"(\d+)시간\s*전", text)
        if match:
            return now - timedelta(hours=int(match.group(1)))

        match = re.search(r"(\d+)일\s*전", text)
        if match:
            return now - timedelta(days=int(match.group(1)))

        match = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", text)
        if match:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        return now
