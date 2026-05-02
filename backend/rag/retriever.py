from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from ingestion.embedder.base import AbstractEmbedder
from models.score import SearchResult
from vectordb.base import AbstractVectorRepository

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(
        self,
        embedder: AbstractEmbedder,
        vector_repo: AbstractVectorRepository,
        top_k: int = 20,
        lookback_days: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._repo = vector_repo
        self._top_k = top_k
        self._lookback_days = lookback_days

    def _cutoff_date(self) -> datetime | None:
        if self._lookback_days is None:
            return None
        return datetime.combine(
            date.today() - timedelta(days=self._lookback_days),
            datetime.min.time(),
        )

    def _filter_by_date(self, results: list[SearchResult]) -> list[SearchResult]:
        cutoff = self._cutoff_date()
        if cutoff is None:
            return results

#       filtered = [r for r in results if r.published_at and r.published_at >= cutoff]
        filtered = [
            r for r in results
            if r.published_at and (r.published_at.replace(tzinfo=None) if r.published_at.tzinfo else r.published_at) >= cutoff
        ]
        
        if len(filtered) < len(results):
            logger.info(
                "시간 필터 적용 — %d건 → %d건 (최근 %d일)",
                len(results), len(filtered), self._lookback_days,
            )
        return filtered

    def retrieve(
        self,
        query: str,
        district_id: str,
        candidate: str | None = None,
    ) -> list[SearchResult]:
        query_vector = self._embedder.embed_query(query)

        filters: dict = {"district_id": district_id}
        if candidate:
            filters["candidate"] = candidate

        raw_results = self._repo.search(
            query_vector=query_vector,
            top_k=self._top_k,
            filters=filters,
        )

        if not raw_results:
            logger.info("필터 검색 결과 0건 — 필터 없이 재검색")
            raw_results = self._repo.search(
                query_vector=query_vector,
                top_k=self._top_k,
                filters=None,
            )

        results: list[SearchResult] = []
        for r in raw_results:
            try:
                results.append(SearchResult(
                    id=r["id"],
                    score=r["score"],
                    text=r.get("text", ""),
                    article_url=r.get("article_url", ""),
                    source=r.get("source", ""),
                    title=r.get("title", ""),
                    published_at=r.get("published_at"),
                    candidate=r.get("candidate", ""),
                    district_id=r.get("district_id", ""),
                ))
            except Exception:
                logger.warning("검색 결과 변환 실패 — id=%s, 스킵", r.get("id"))
                continue

        results = self._filter_by_date(results)

        logger.info(
            "검색 완료 — query='%s', district=%s, candidate=%s → %d건",
            query[:30], district_id, candidate or "전체", len(results),
        )
        for r in results:
            logger.debug(
                "  [%s] score=%.4f | %s | %s",
                r.id[:8], r.score, r.title[:40], r.published_at,
            )
        return results

    def retrieve_for_district(self, district: dict) -> list[SearchResult]:
        """선거구의 모든 후보에 대해 검색하여 통합 결과 반환."""
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for cand in district.get("candidates", []):
            name = cand["name"]
            query = f"{district['name']} {name} 선거 판세"
            results = self.retrieve(
                query=query,
                district_id=district["id"],
                candidate=name,
            )
            for r in results:
                if r.id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.id)

        logger.info(
            "선거구 통합 검색 완료 — %s, 후보 %d명 → 총 %d건 (중복 제거)",
            district["name"], len(district.get("candidates", [])), len(all_results),
        )
        return all_results
