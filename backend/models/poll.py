from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class PollEntry(BaseModel):
    """여론조사 개별 항목 — 조사기관·날짜·선거구·후보별 지지율."""

    id: str = ""
    district_id: str
    candidate: str
    party: str
    support: float = Field(ge=0, le=100)
    pollster: str
    survey_date: date
    created_at: datetime = Field(default_factory=datetime.now)


class PollCandidateSupport(BaseModel):
    candidate: str
    party: str
    support: float


class PollSummary(BaseModel):
    """선거구별 적용 중인 여론조사 요약 (Scorer 프롬프트 주입용)."""

    district_id: str
    pollster: str
    survey_date: date
    candidates: list[PollCandidateSupport]
