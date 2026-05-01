from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CandidateScoreResponse(BaseModel):
    candidate: str
    party: str
    verdict: str
    win_probability: float
    reasoning: str


class VerdictResponse(BaseModel):
    district_id: str
    district_name: str
    date: datetime
    candidates: list[CandidateScoreResponse]
    total_chunks_analyzed: int
    summary: str


class VerdictListResponse(BaseModel):
    district_id: str
    count: int
    verdicts: list[VerdictResponse]


class TimeSeriesPoint(BaseModel):
    date: datetime
    candidates: dict[str, float]


class TimeSeriesResponse(BaseModel):
    district_id: str
    district_name: str
    points: list[TimeSeriesPoint]


class VerdictRunRequest(BaseModel):
    district_id: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    lookback_days: int | None = Field(default=None, ge=1, le=365)
    skip_score: bool = False
