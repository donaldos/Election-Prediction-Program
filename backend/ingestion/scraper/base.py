from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import date, timedelta

from ingestion.base_registry import ComponentRegistry
from ingestion.scraper.url_store import ScrapedUrlStore
from models.article import RawArticle

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 2

_BODY_SELECTORS = [
    "#dic_area",
    "#articleBodyContents",
    ".article_body",
    "#articeBody",
    "#article-body",
    ".news_cnt_detail_wrap",
    ".article-body",
    "#article_body",
    ".article_txt",
    ".newsct_article",
    "article",
]


def fetch_article_body(url: str, *, headers: dict | None = None, timeout: float = 10) -> str:
    """기사 상세 페이지에서 본문 전문을 추출한다.

    여러 매체의 셀렉터를 순차 시도하고, 실패 시 og:description을 fallback으로 사용.
    네트워크 오류 시 빈 문자열 반환.
    """
    import httpx
    from bs4 import BeautifulSoup

    try:
        resp = httpx.get(url, headers=headers or {}, timeout=timeout, follow_redirects=True)
        if resp.status_code >= 400:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        logger.debug("기사 본문 요청 실패: %s", url)
        return ""

    for sel in _BODY_SELECTORS:
        tag = soup.select_one(sel)
        if not tag:
            continue
        for unwanted in tag.select("script, style, iframe, .reporter_area, .copyright, .article_relate"):
            unwanted.decompose()
        text = tag.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > 80:
            return text

    og = soup.select_one('meta[property="og:description"]')
    if og:
        desc = og.get("content", "")
        if len(desc) > 50:
            return desc

    return ""


class AbstractScraper(ABC):

    @abstractmethod
    def scrape(
        self,
        keywords: list[str],
        date_from: date | None = None,
        date_to: date | None = None,
        max_articles: int = 50,
    ) -> list[RawArticle]:
        """키워드로 뉴스를 검색·수집하여 RawArticle 리스트 반환.

        date_from/date_to가 None이면 lookback_days 기준으로 자동 설정.
        실패 시 빈 리스트 반환. 예외 raise 금지.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...

    @property
    def request_delay_sec(self) -> float:
        return 1.5

    @property
    def lookback_days(self) -> int:
        return DEFAULT_LOOKBACK_DAYS

    def resolve_date_range(
        self, date_from: date | None, date_to: date | None
    ) -> tuple[date, date]:
        """date_from/date_to가 None이면 오늘 기준 lookback_days일 전~오늘로 설정."""
        today = date.today()
        resolved_to = date_to or today
        resolved_from = date_from or (today - timedelta(days=self.lookback_days))
        return resolved_from, resolved_to


ScraperRegistry = ComponentRegistry(AbstractScraper, "Scraper")
