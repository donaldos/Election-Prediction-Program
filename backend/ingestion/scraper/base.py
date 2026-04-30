from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta

from ingestion.base_registry import ComponentRegistry
from ingestion.scraper.url_store import ScrapedUrlStore
from models.article import RawArticle

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 2


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
