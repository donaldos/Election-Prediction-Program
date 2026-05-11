from __future__ import annotations

import logging

from models.score import SearchResult

logger = logging.getLogger(__name__)


class Reranker:

    def __init__(
        self,
        min_score: float = 0.3,
        deduplicate: bool = True,
    ) -> None:
        self._min_score = min_score
        self._deduplicate = deduplicate

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            logger.warning("재정렬 입력이 비어 있음 — 빈 리스트 반환")
            return []

        before_count = len(results)

        filtered = [r for r in results if r.score >= self._min_score]
        score_dropped = before_count - len(filtered)
        if score_dropped:
            logger.debug("임계값 필터링 — %d건 제거 (min_score=%.2f)", score_dropped, self._min_score)

        if self._deduplicate:
            before_dedup = len(filtered)
            filtered = self._deduplicate_by_url(filtered)
            dedup_dropped = before_dedup - len(filtered)
            if dedup_dropped:
                logger.debug("URL 중복 제거 — %d건 제거", dedup_dropped)

        filtered.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "재정렬 완료 — %d건 → %d건 (min_score=%.2f, dedup=%s)",
            before_count, len(filtered), self._min_score, self._deduplicate,
        )
        return filtered

    def rerank_grouped(
        self,
        grouped: dict[str, dict[str, list[SearchResult]]],
    ) -> dict[str, dict[str, list[SearchResult]]]:
        """그룹별(후보 → 분석 항목)로 재정렬 및 필터링."""
        result: dict[str, dict[str, list[SearchResult]]] = {}
        total_before = 0
        total_after = 0

        for group_key, categories in grouped.items():
            result[group_key] = {}
            for category, chunks in categories.items():
                total_before += len(chunks)
                filtered = [r for r in chunks if r.score >= self._min_score]
                if self._deduplicate:
                    filtered = self._deduplicate_by_url(filtered)
                filtered.sort(key=lambda r: r.score, reverse=True)
                total_after += len(filtered)
                if filtered:
                    result[group_key][category] = filtered

        logger.info(
            "그룹별 재정렬 완료 — %d건 → %d건 (min_score=%.2f, dedup=%s)",
            total_before, total_after, self._min_score, self._deduplicate,
        )
        return result

    @staticmethod
    def _deduplicate_by_url(results: list[SearchResult]) -> list[SearchResult]:
        best_by_url: dict[str, SearchResult] = {}
        for r in results:
            if r.article_url not in best_by_url or r.score > best_by_url[r.article_url].score:
                best_by_url[r.article_url] = r
        return list(best_by_url.values())
