from __future__ import annotations

import logging
import math

from models.score import SearchResult

logger = logging.getLogger(__name__)


class Reranker:

    def __init__(
        self,
        min_score: float = 0.3,
        deduplicate: bool = True,
        cross_encoder_model: str | None = None,
        cross_encoder_top_n: int | None = None,
    ) -> None:
        self._min_score = min_score
        self._deduplicate = deduplicate
        self._cross_encoder_model = cross_encoder_model
        self._cross_encoder_top_n = cross_encoder_top_n
        self._cross_encoder = None

    def _load_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        if self._cross_encoder_model is None:
            return None
        from sentence_transformers import CrossEncoder
        logger.info("Cross-encoder 로딩 — %s", self._cross_encoder_model)
        self._cross_encoder = CrossEncoder(self._cross_encoder_model)
        logger.info("Cross-encoder 로딩 완료")
        return self._cross_encoder

    def _cross_encode_rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        ce = self._load_cross_encoder()
        if ce is None or not chunks:
            return chunks

        effective_top_n = min(
            top_n or self._cross_encoder_top_n or len(chunks),
            len(chunks),
        )

        pairs = [(query, c.text) for c in chunks]
        raw_scores = ce.predict(pairs)

        scored = sorted(
            zip(chunks, raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result: list[SearchResult] = []
        for chunk, ce_score in scored[:effective_top_n]:
            norm_score = 1.0 / (1.0 + math.exp(-float(ce_score)))
            result.append(chunk.model_copy(update={"score": norm_score}))

        logger.debug(
            "Cross-encoder 재정렬 — query='%s', %d건 → %d건",
            query[:30], len(chunks), len(result),
        )
        return result

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

        if self._cross_encoder_model:
            filtered = self._cross_encode_rerank(query, filtered)

        logger.info(
            "재정렬 완료 — %d건 → %d건 (min_score=%.2f, dedup=%s, cross_encoder=%s)",
            before_count, len(filtered), self._min_score, self._deduplicate,
            bool(self._cross_encoder_model),
        )
        return filtered

    def rerank_grouped(
        self,
        grouped: dict[str, dict[str, list[SearchResult]]],
        district_name: str = "",
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

                if self._cross_encoder_model and filtered:
                    if group_key == "_common":
                        query = f"{district_name} {category}"
                    else:
                        query = f"{group_key} {category}"
                    filtered = self._cross_encode_rerank(query, filtered)

                total_after += len(filtered)
                if filtered:
                    result[group_key][category] = filtered

        logger.info(
            "그룹별 재정렬 완료 — %d건 → %d건 (min_score=%.2f, dedup=%s, cross_encoder=%s)",
            total_before, total_after, self._min_score, self._deduplicate,
            bool(self._cross_encoder_model),
        )
        return result

    @staticmethod
    def _deduplicate_by_url(results: list[SearchResult]) -> list[SearchResult]:
        best_by_url: dict[str, SearchResult] = {}
        for r in results:
            if r.article_url not in best_by_url or r.score > best_by_url[r.article_url].score:
                best_by_url[r.article_url] = r
        return list(best_by_url.values())
