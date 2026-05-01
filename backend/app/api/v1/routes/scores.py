"""판세 결과 API — 일반 사용자 + 관리자 판정 실행."""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.schemas.score import (
    CandidateScoreResponse,
    TimeSeriesPoint,
    TimeSeriesResponse,
    VerdictListResponse,
    VerdictResponse,
    VerdictRunRequest,
)
from app.core.dependencies import get_config, reload_config
from app.core.pipeline_runner import pipeline_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scores", tags=["scores"])


def _verdict_to_response(v) -> VerdictResponse:
    return VerdictResponse(
        district_id=v.district_id,
        district_name=v.district_name,
        date=v.date,
        candidates=[
            CandidateScoreResponse(
                candidate=c.candidate,
                party=c.party,
                verdict=c.verdict,
                win_probability=c.win_probability,
                reasoning=c.reasoning,
            )
            for c in v.candidates
        ],
        total_chunks_analyzed=v.total_chunks_analyzed,
        summary=v.summary,
    )


# ── 최신 판정 결과 ─────────────────────────────────

@router.get("/{district_id}/latest", response_model=VerdictResponse)
def get_latest_verdict(district_id: str):
    from rag.verdict_store import VerdictStore

    store = VerdictStore()
    verdict = store.load_latest(district_id)
    if not verdict:
        raise HTTPException(status_code=404, detail=f"'{district_id}' 판정 결과가 없습니다.")
    return _verdict_to_response(verdict)


# ── 판정 이력 조회 ─────────────────────────────────

@router.get("/{district_id}/history", response_model=VerdictListResponse)
def get_verdict_history(
    district_id: str,
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    from rag.verdict_store import VerdictStore

    store = VerdictStore()
    verdicts = store.load_range(district_id, date_from=date_from, date_to=date_to)
    verdicts = verdicts[-limit:]

    return VerdictListResponse(
        district_id=district_id,
        count=len(verdicts),
        verdicts=[_verdict_to_response(v) for v in verdicts],
    )


# ── 시계열 데이터 (차트용) ─────────────────────────

@router.get("/{district_id}/timeseries", response_model=TimeSeriesResponse)
def get_timeseries(
    district_id: str,
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
):
    from rag.verdict_store import VerdictStore

    config = get_config()
    district = None
    for d in config.get("districts", []):
        if d["id"] == district_id:
            district = d
            break
    if not district:
        raise HTTPException(status_code=404, detail=f"선거구 '{district_id}'를 찾을 수 없습니다.")

    store = VerdictStore()
    verdicts = store.load_range(district_id, date_from=date_from, date_to=date_to)

    points = [
        TimeSeriesPoint(
            date=v.date,
            candidates={c.candidate: c.win_probability for c in v.candidates},
        )
        for v in verdicts
    ]

    return TimeSeriesResponse(
        district_id=district_id,
        district_name=district["name"],
        points=points,
    )


# ── 판정 실행 (관리자) ─────────────────────────────

@router.post("/{district_id}/run", response_model=VerdictResponse)
def run_verdict(district_id: str, req: VerdictRunRequest | None = None):
    config = reload_config()

    district = None
    for d in config.get("districts", []):
        if d["id"] == district_id:
            district = d
            break
    if not district:
        raise HTTPException(status_code=404, detail=f"선거구 '{district_id}'를 찾을 수 없습니다.")

    if pipeline_runner.is_running:
        raise HTTPException(status_code=409, detail="파이프라인이 이미 실행 중입니다.")

    from dotenv import load_dotenv
    load_dotenv()

    if req and req.top_k:
        config.setdefault("rag", {}).setdefault("retriever", {})["top_k"] = req.top_k
    if req and req.lookback_days:
        config.setdefault("rag", {}).setdefault("retriever", {})["lookback_days"] = req.lookback_days

    from rag.pipeline import _build_embedder, _build_retriever, _build_reranker, _build_scorer, _build_vector_repo

    embedder = _build_embedder(config)
    vector_repo = _build_vector_repo(config)
    retriever = _build_retriever(config, embedder, vector_repo)
    reranker = _build_reranker(config)

    all_results = retriever.retrieve_for_district(district)
    query = f"{district['name']} 선거 판세"
    all_results = reranker.rerank(query, all_results)

    if req and req.skip_score:
        raise HTTPException(status_code=400, detail="skip_score=true — 판정 결과를 생성할 수 없습니다.")

    scorer = _build_scorer(config)
    verdict = scorer.score(all_results, district)

    from rag.verdict_store import VerdictStore
    store = VerdictStore()
    store.save(verdict)

    return _verdict_to_response(verdict)
