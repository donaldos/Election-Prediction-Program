from __future__ import annotations

import logging

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
    ) -> None:
        self._embedder = embedder
        self._repo = vector_repo
        self._top_k = top_k

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

        logger.info(
            "검색 완료 — query='%s', district=%s, candidate=%s → %d건",
            query[:30], district_id, candidate or "전체", len(results),
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
