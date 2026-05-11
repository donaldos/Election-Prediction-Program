from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from ingestion.embedder.base import AbstractEmbedder
from models.score import SearchResult
from vectordb.base import AbstractVectorRepository

logger = logging.getLogger(__name__)

QUERY_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "config" / "query_templates.json"

_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("지지율 추이", "지지율 추이"),
    ("출마 여론", "출마 여론"),
    ("선거 전략", "선거 전략"),
    ("공약", "공약 반응"),
    ("지지율", "지지율"),
    ("강점", "강점"),
    ("약점", "약점"),
    ("이슈", "이슈"),
    ("판세", "판세"),
    ("여론조사", "여론조사"),
]


def _detect_category(query: str) -> str:
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in query:
            return category
    return "기타"


def _load_query_templates() -> dict:
    if not QUERY_TEMPLATES_PATH.exists():
        return {}
    with QUERY_TEMPLATES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class Retriever:

    def __init__(
        self,
        embedder: AbstractEmbedder,
        vector_repo: AbstractVectorRepository,
        top_k: int = 20,
        top_k_common: int | None = None,
        top_k_candidate: int | None = None,
        lookback_days: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._repo = vector_repo
        self._top_k = top_k
        self._top_k_common = top_k_common or top_k
        self._top_k_candidate = top_k_candidate or top_k
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
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        effective_top_k = top_k or self._top_k
        logger.info("━" * 50)
        logger.info("의미 검색 시작 — query='%s'", query)
        logger.info("  필터: district_id=%s, candidate=%s", district_id, candidate or "(전체)")
        logger.info("  파라미터: top_k=%d, lookback_days=%s", effective_top_k, self._lookback_days or "전체")

        query_vector = self._embedder.embed_query(query)
        logger.info("  쿼리 임베딩 완료 — 벡터 차원=%d", len(query_vector))

        filters: dict = {"district_id": district_id}
        if candidate:
            filters["candidate"] = candidate

        raw_results = self._repo.search(
            query_vector=query_vector,
            top_k=effective_top_k,
            filters=filters,
        )
        logger.info("  VectorDB 검색 결과: %d건 (필터 적용)", len(raw_results))

        if not raw_results:
            logger.info("  필터 검색 결과 0건 — 필터 없이 재검색")
            raw_results = self._repo.search(
                query_vector=query_vector,
                top_k=effective_top_k,
                filters=None,
            )
            logger.info("  VectorDB 재검색 결과: %d건 (필터 없음)", len(raw_results))

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
                logger.warning("  검색 결과 변환 실패 — id=%s, 스킵", r.get("id"))
                continue

        before_filter = len(results)
        results = self._filter_by_date(results)

        logger.info("  변환 완료: %d건 → 시간 필터 후 %d건", before_filter, len(results))

        for i, r in enumerate(results, 1):
            logger.info(
                "  [%d] score=%.4f | %s | %s | %s",
                i, r.score, r.title[:50], r.source, r.published_at,
            )

        logger.info("검색 완료 — query='%s' → 최종 %d건", query[:30], len(results))
        return results

    def retrieve_for_district(self, district: dict) -> list[SearchResult]:
        """선거구의 모든 후보에 대해 검색하여 통합 결과 반환."""
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        templates = _load_query_templates()
        district_templates = templates.get(district["id"], {})
        common_queries = district_templates.get("_common", [])

        logger.info("=" * 60)
        logger.info("선거구 통합 검색 시작 — %s (후보 %d명)", district["name"], len(district.get("candidates", [])))
        if district_templates:
            logger.info("  쿼리 템플릿 사용: query_templates.json")
        logger.info("=" * 60)

        def _add_results(results: list[SearchResult]) -> int:
            new_count = 0
            for r in results:
                if r.id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.id)
                    new_count += 1
            return new_count

        logger.info(
            "  top_k: common=%d, candidate=%d",
            self._top_k_common, self._top_k_candidate,
        )

        if common_queries:
            logger.info("")
            logger.info("▶ 공통 쿼리 (%d건, top_k=%d)", len(common_queries), self._top_k_common)
            for query in common_queries:
                results = self.retrieve(
                    query=query,
                    district_id=district["id"],
                    candidate=None,
                    top_k=self._top_k_common,
                )
                new_count = _add_results(results)
                logger.info("  신규 %d건 추가 (중복 %d건 제외)", new_count, len(results) - new_count)

        for cand in district.get("candidates", []):
            name = cand["name"]
            party = cand.get("party", "")

            candidate_queries = district_templates.get(name)
            if candidate_queries is None:
                candidate_queries = [f"{district['name']} {name} 선거 판세"]

            logger.info("")
            logger.info("▶ 후보: %s (%s) — 쿼리 %d건 (top_k=%d)", name, party, len(candidate_queries), self._top_k_candidate)

            for query in candidate_queries:
                results = self.retrieve(
                    query=query,
                    district_id=district["id"],
                    candidate=name,
                    top_k=self._top_k_candidate,
                )
                new_count = _add_results(results)
                logger.info("  신규 %d건 추가 (중복 %d건 제외)", new_count, len(results) - new_count)

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "선거구 통합 검색 완료 — %s, 후보 %d명 → 총 %d건 (중복 제거)",
            district["name"], len(district.get("candidates", [])), len(all_results),
        )
        logger.info("=" * 60)
        return all_results

    def retrieve_for_district_grouped(
        self, district: dict,
    ) -> dict[str, dict[str, list[SearchResult]]]:
        """선거구의 모든 후보에 대해 분석 항목별로 그룹핑된 검색 결과 반환."""
        grouped: dict[str, dict[str, list[SearchResult]]] = {}

        templates = _load_query_templates()
        district_templates = templates.get(district["id"], {})
        common_queries = district_templates.get("_common", [])

        logger.info("=" * 60)
        logger.info(
            "선거구 그룹별 검색 시작 — %s (후보 %d명)",
            district["name"], len(district.get("candidates", [])),
        )
        logger.info("=" * 60)
        logger.info(
            "  top_k: common=%d, candidate=%d",
            self._top_k_common, self._top_k_candidate,
        )

        if common_queries:
            grouped["_common"] = {}
            logger.info("")
            logger.info("▶ 공통 쿼리 (%d건, top_k=%d)", len(common_queries), self._top_k_common)
            for query in common_queries:
                category = _detect_category(query)
                results = self.retrieve(
                    query=query,
                    district_id=district["id"],
                    candidate=None,
                    top_k=self._top_k_common,
                )
                grouped["_common"][category] = results
                logger.info("  [%s] %d건", category, len(results))

        for cand in district.get("candidates", []):
            name = cand["name"]
            party = cand.get("party", "")

            candidate_queries = district_templates.get(name)
            if candidate_queries is None:
                candidate_queries = [f"{district['name']} {name} 선거 판세"]

            grouped[name] = {}
            logger.info("")
            logger.info("▶ 후보: %s (%s) — 쿼리 %d건 (top_k=%d)", name, party, len(candidate_queries), self._top_k_candidate)

            for query in candidate_queries:
                category = _detect_category(query)
                results = self.retrieve(
                    query=query,
                    district_id=district["id"],
                    candidate=name,
                    top_k=self._top_k_candidate,
                )
                grouped[name][category] = results
                logger.info("  [%s] %d건", category, len(results))

        total = sum(
            len(chunks)
            for categories in grouped.values()
            for chunks in categories.values()
        )
        unique_ids = {
            c.id
            for categories in grouped.values()
            for chunks in categories.values()
            for c in chunks
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "선거구 그룹별 검색 완료 — %s, 총 %d건 (고유 %d건)",
            district["name"], total, len(unique_ids),
        )
        logger.info("=" * 60)
        return grouped
