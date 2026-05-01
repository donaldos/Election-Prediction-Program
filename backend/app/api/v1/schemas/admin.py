from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ── 파이프라인 실행 ─────────────────────────────────

class PipelineRunRequest(BaseModel):
    scraper: str = Field(default="all", pattern="^(naver|political|all)$")
    days: int | None = Field(default=None, ge=1, le=365)
    skip_chunk: bool = False
    skip_embed: bool = False
    skip_store: bool = False


class PipelineRunResponse(BaseModel):
    task_id: str
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    task_id: str | None
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict | None = None


# ── VectorDB 관리 ──────────────────────────────────

class VectorDBStatsResponse(BaseModel):
    type: str
    collection: str
    count: int


class PurgeRequest(BaseModel):
    purge_days: int = Field(ge=1, le=365)


class PurgeResponse(BaseModel):
    deleted: int
    remaining: int
    message: str


class RebuildResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ── 설정 조회/변경 ─────────────────────────────────

class RAGConfigResponse(BaseModel):
    retriever: dict
    reranker: dict
    scorer: dict
    purge_days: int | None


class RAGConfigUpdateRequest(BaseModel):
    lookback_days: int | None = Field(default=None, ge=1, le=365)
    top_k: int | None = Field(default=None, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    purge_days: int | None = Field(default=None, ge=1, le=365)
    scorer_provider: str | None = Field(default=None, pattern="^(openai|anthropic)$")
    scorer_model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ConfigResponse(BaseModel):
    districts: list[dict]
    scrapers: dict
    chunker: dict
    embedder: dict
    vectordb: dict
    rag: dict


# ── 선거구 ─────────────────────────────────────────

class DistrictResponse(BaseModel):
    id: str
    name: str
    candidates: list[dict]
