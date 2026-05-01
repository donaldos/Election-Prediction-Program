"""관리자 API 라우터 — 파이프라인 실행, VectorDB 관리, 설정 조회/변경."""
from __future__ import annotations

import copy
import logging

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas.admin import (
    ConfigResponse,
    DistrictResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
    PurgeRequest,
    PurgeResponse,
    RAGConfigResponse,
    RAGConfigUpdateRequest,
    RebuildResponse,
    VectorDBStatsResponse,
)
from app.core.dependencies import get_config, get_vector_repo, reload_config, save_config
from app.core.pipeline_runner import pipeline_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── 파이프라인 실행 제어 ────────────────────────────

@router.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(req: PipelineRunRequest):
    config = reload_config()
    try:
        task = pipeline_runner.run_pipeline(
            config,
            scraper_name=req.scraper,
            days=req.days,
            skip_chunk=req.skip_chunk,
            skip_embed=req.skip_embed,
            skip_store=req.skip_store,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return PipelineRunResponse(
        task_id=task.task_id,
        status=task.status.value,
        message="파이프라인 실행이 시작되었습니다.",
    )


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
def get_pipeline_status():
    task = pipeline_runner.current_task
    if task is None:
        return PipelineStatusResponse(task_id=None, status="idle")

    return PipelineStatusResponse(
        task_id=task.task_id,
        status=task.status.value,
        started_at=task.started_at,
        finished_at=task.finished_at,
        result=task.result or ({"error": task.error} if task.error else None),
    )


@router.post("/pipeline/rebuild", response_model=RebuildResponse)
def rebuild_vectordb():
    config = reload_config()
    try:
        task = pipeline_runner.run_rebuild(config)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return RebuildResponse(
        task_id=task.task_id,
        status=task.status.value,
        message="VectorDB 재구축이 시작되었습니다. 기존 데이터를 삭제 후 전체 파이프라인을 실행합니다.",
    )


# ── VectorDB 관리 ──────────────────────────────────

@router.get("/vectordb/stats", response_model=VectorDBStatsResponse)
def vectordb_stats():
    config = get_config()
    repo = get_vector_repo(config)
    cfg = config.get("vectordb", {})
    return VectorDBStatsResponse(
        type=cfg.get("type", "chroma"),
        collection=cfg.get("collection", "election_chunks"),
        count=repo.count(),
    )


@router.post("/vectordb/purge", response_model=PurgeResponse)
def purge_vectordb(req: PurgeRequest):
    config = get_config()
    repo = get_vector_repo(config)
    before = repo.count()
    deleted = repo.delete_older_than(req.purge_days)
    remaining = repo.count()

    return PurgeResponse(
        deleted=deleted,
        remaining=remaining,
        message=f"{req.purge_days}일 이전 벡터 {deleted}개 삭제 완료. 잔여 {remaining}개.",
    )


# ── 설정 조회/변경 ─────────────────────────────────

@router.get("/config", response_model=ConfigResponse)
def get_full_config():
    config = reload_config()
    return ConfigResponse(
        districts=config.get("districts", []),
        scrapers=config.get("scrapers", {}),
        chunker=config.get("chunker", {}),
        embedder=config.get("embedder", {}),
        vectordb=config.get("vectordb", {}),
        rag=config.get("rag", {}),
    )


@router.get("/config/rag", response_model=RAGConfigResponse)
def get_rag_config():
    config = reload_config()
    rag = config.get("rag", {})
    return RAGConfigResponse(
        retriever=rag.get("retriever", {}),
        reranker=rag.get("reranker", {}),
        scorer=rag.get("scorer", {}),
        purge_days=rag.get("purge_days"),
    )


@router.patch("/config/rag", response_model=RAGConfigResponse)
def update_rag_config(req: RAGConfigUpdateRequest):
    config = reload_config()
    config = copy.deepcopy(config)
    rag = config.setdefault("rag", {})

    if req.lookback_days is not None:
        rag.setdefault("retriever", {})["lookback_days"] = req.lookback_days
    if req.top_k is not None:
        rag.setdefault("retriever", {})["top_k"] = req.top_k
    if req.min_score is not None:
        rag.setdefault("reranker", {})["min_score"] = req.min_score
    if req.purge_days is not None:
        rag["purge_days"] = req.purge_days
    if req.scorer_provider is not None:
        rag.setdefault("scorer", {})["provider"] = req.scorer_provider
    if req.scorer_model is not None:
        rag.setdefault("scorer", {})["model"] = req.scorer_model
    if req.temperature is not None:
        rag.setdefault("scorer", {})["temperature"] = req.temperature

    save_config(config)
    logger.info("RAG 설정 변경 완료: %s", req.model_dump(exclude_none=True))

    return RAGConfigResponse(
        retriever=rag.get("retriever", {}),
        reranker=rag.get("reranker", {}),
        scorer=rag.get("scorer", {}),
        purge_days=rag.get("purge_days"),
    )


# ── 선거구 조회 ────────────────────────────────────

@router.get("/districts", response_model=list[DistrictResponse])
def list_districts():
    config = get_config()
    return [
        DistrictResponse(
            id=d["id"],
            name=d["name"],
            candidates=d.get("candidates", []),
        )
        for d in config.get("districts", [])
    ]
