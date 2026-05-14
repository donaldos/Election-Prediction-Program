from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
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
    "Referer": "https://www.naver.com",
}

SEARCH_URL = "https://search.naver.com/search.naver"

SEL_TITLE = 'a[data-heatmap-target=".tit"]'
SEL_SUMMARY = 'a[data-heatmap-target=".body"]'
SEL_PRESS = 'span.sds-comps-profile-info-title-text'
SEL_DATE = 'span.sds-comps-profile-info-subtext'


@ScraperRegistry.register("naver")
class NaverNewsScraper(AbstractScraper):

    def __init__(
        self,
        max_articles_per_run: int = 50,
        request_delay_sec: float = 1.5,
        lookback_days: int = 2,
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
        return "naver_news"

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
            "[%s] 수집 시작 — keywords=%s, 기간=%s~%s, limit=%d",
            self.source_name, keywords, resolved_from, resolved_to, limit,
        )

        articles: list[RawArticle] = []
        seen_urls: set[str] = set()
        skipped_existing = 0

        try:
            for keyword in keywords:
                if len(articles) >= limit:
                    logger.info("[%s] limit 도달 — 키워드 루프 중단", self.source_name)
                    break

                logger.info("[%s] 키워드 검색: '%s'", self.source_name, keyword)
                fetched = self._search_keyword(keyword, resolved_from, resolved_to, limit - len(articles))

                for article in fetched:
                    if self._url_store.contains(article.url):
                        skipped_existing += 1
                        continue
                    if article.url not in seen_urls:
                        seen_urls.add(article.url)
                        articles.append(article)

                logger.info(
                    "[%s] '%s' 검색 결과: %d건 수집, 누적 %d건",
                    self.source_name, keyword, len(fetched), len(articles),
                )
                time.sleep(self.request_delay_sec)
        except Exception as e:
            logger.warning("[%s] 수집 중 오류 발생: %s", self.source_name, e)

        result = articles[:limit]
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

    def _search_keyword(
        self,
        keyword: str,
        date_from: date,
        date_to: date,
        remaining: int,
    ) -> list[RawArticle]:
        import httpx
        from bs4 import BeautifulSoup

        articles: list[RawArticle] = []
        offset = 1
        page = 0

        while len(articles) < remaining:
            page += 1
            params = {
                "where": "news",
                "query": keyword,
                "ds": date_from.strftime("%Y.%m.%d"),
                "de": date_to.strftime("%Y.%m.%d"),
                "sort": "1",
                "start": str(offset),
            }

            logger.debug("[%s] HTTP 요청 — keyword='%s', page=%d, offset=%d", self.source_name, keyword, page, offset)
            resp = httpx.get(SEARCH_URL, params=params, headers=HEADERS, timeout=10)

            if resp.status_code in (404, 429):
                logger.warning(
                    "[%s] HTTP %d 응답 — 검색 중단 (keyword='%s', page=%d)",
                    self.source_name, resp.status_code, keyword, page,
                )
                break
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            title_tags = soup.select(SEL_TITLE)
            if not title_tags:
                logger.debug("[%s] 검색 결과 없음 — 페이지네이션 종료 (keyword='%s', page=%d)", self.source_name, keyword, page)
                break

            page_count = 0
            for title_tag in title_tags:
                container = self._find_article_container(title_tag)
                if not container:
                    continue

                url = title_tag.get("href", "")
                title = title_tag.get_text(strip=True)
                summary = ""
                summary_tag = container.select_one(SEL_SUMMARY)
                if summary_tag:
                    summary = summary_tag.get_text(strip=True)

                published_at = self._parse_date(container)
                page_count += 1

                articles.append(
                    RawArticle(
                        url=url,
                        source=self.source_name,
                        title=title,
                        body=summary,
                        published_at=published_at,
                        matched_keywords=[keyword],
                    )
                )

            logger.debug("[%s] page=%d 파싱 완료 — %d건 (누적 %d건)", self.source_name, page, page_count, len(articles))
            offset += 10
            time.sleep(self.request_delay_sec)

        return articles

    @staticmethod
    def _find_article_container(title_tag) -> object | None:
        """`.tit` 링크에서 위로 올라가며 `.body`와 `.prof`를 모두 포함하는 컨테이너를 찾는다."""
        container = title_tag.find_parent("div")
        while container:
            has_body = container.select_one(SEL_SUMMARY)
            has_prof = container.select_one('a[data-heatmap-target=".prof"]')
            if has_body and has_prof:
                return container
            container = container.find_parent("div")
        return None

    def _parse_date(self, item) -> datetime:
        from bs4 import Tag
        from datetime import timedelta

        now = datetime.now()
        info_tags = item.select(SEL_DATE)
        for tag in info_tags:
            if not isinstance(tag, Tag):
                continue
            text = tag.get_text(strip=True)

            match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})\.", text)
            if match:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))

            match = re.search(r"(\d+)분\s*전", text)
            if match:
                return now - timedelta(minutes=int(match.group(1)))

            match = re.search(r"(\d+)시간\s*전", text)
            if match:
                return now - timedelta(hours=int(match.group(1)))

            match = re.search(r"(\d+)일\s*전", text)
            if match:
                return now - timedelta(days=int(match.group(1)))

        logger.debug("[%s] 날짜 파싱 실패 — 현�� 시각으로 대체", self.source_name)
        return now
