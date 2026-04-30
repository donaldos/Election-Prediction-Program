"""기사 → 후보/선거구 자동 태깅.

수집된 기사의 제목+본문에서 config.yaml의 후보 키워드를 매칭하여
candidate, district_id, matched_keywords 필드를 채운다.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from models.article import RawArticle

logger = logging.getLogger(__name__)


def _count_keyword_hits(text: str, keywords: list[str]) -> dict[str, int]:
    hits: dict[str, int] = {}
    for kw in keywords:
        count = text.count(kw)
        if count > 0:
            hits[kw] = count
    return hits


def tag_articles(
    articles: list[RawArticle],
    districts: list[dict],
) -> list[RawArticle]:
    """기사 목록에 candidate / district_id / matched_keywords를 태깅한다.

    매칭 규칙:
    - 제목+본문에서 각 후보의 keywords 출현 횟수를 센다.
    - 한 선거구의 후보 1명만 매칭 → candidate + district_id 태깅
    - 한 선거구의 후보 여러 명 매칭 → district_id만 태깅 (비교 기사)
    - 여러 선거구 매칭 → 키워드 출현 합계가 큰 선거구 우선
    - 매칭 없음 → 태깅하지 않음 (빈 문자열 유지)
    """
    candidate_map = _build_candidate_map(districts)
    tagged_count = 0

    for article in articles:
        if article.candidate and article.district_id:
            continue

        search_text = f"{article.title} {article.body}"
        district_id, candidate, matched = _match_article(search_text, candidate_map)

        if district_id:
            article.district_id = district_id
            article.candidate = candidate
            article.matched_keywords = matched
            tagged_count += 1

    untagged = len(articles) - tagged_count
    logger.info(
        "태깅 완료 — 전체 %d건, 태깅 %d건, 미태깅 %d건",
        len(articles), tagged_count, untagged,
    )
    return articles


def _build_candidate_map(districts: list[dict]) -> dict[str, list[dict]]:
    """district_id → [{name, party, keywords}, ...] 매핑."""
    result: dict[str, list[dict]] = {}
    for d in districts:
        result[d["id"]] = [
            {
                "name": c["name"],
                "party": c["party"],
                "keywords": c.get("keywords", []),
            }
            for c in d.get("candidates", [])
        ]
    return result


def _match_article(
    text: str,
    candidate_map: dict[str, list[dict]],
) -> tuple[str, str, list[str]]:
    """기사 텍스트에서 가장 적합한 (district_id, candidate, matched_keywords)를 반환."""
    district_scores: dict[str, dict] = {}

    for district_id, candidates in candidate_map.items():
        matched_candidates: list[dict] = []

        for cand in candidates:
            hits = _count_keyword_hits(text, cand["keywords"])
            if hits:
                matched_candidates.append({
                    "name": cand["name"],
                    "hits": hits,
                    "total": sum(hits.values()),
                })

        if matched_candidates:
            total_hits = sum(c["total"] for c in matched_candidates)
            all_keywords = []
            for c in matched_candidates:
                all_keywords.extend(c["hits"].keys())

            district_scores[district_id] = {
                "total_hits": total_hits,
                "matched_candidates": matched_candidates,
                "all_keywords": all_keywords,
            }

    if not district_scores:
        return "", "", []

    best_district = max(district_scores, key=lambda d: district_scores[d]["total_hits"])
    info = district_scores[best_district]
    matched_candidates = info["matched_candidates"]

    if len(matched_candidates) == 1:
        candidate = matched_candidates[0]["name"]
    else:
        candidate = ""

    return best_district, candidate, info["all_keywords"]
