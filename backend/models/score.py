from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SearchResult(BaseModel):
    """VectorDB 검색 결과 단위. Retriever 출력."""

    id: str
    score: float
    text: str
    article_url: str
    source: str
    title: str
    published_at: datetime
    candidate: str
    district_id: str


class CandidateScore(BaseModel):
    """후보별 판정 결과. Scorer 출력."""

    candidate: str
    party: str
    district_id: str

    verdict: str
    win_probability: float
    reasoning: str

    supporting_chunks: list[str]
    chunk_count: int


class DailyVerdict(BaseModel):
    """선거구별 일일 판정 결과. API 응답 단위."""

    district_id: str
    district_name: str
    date: datetime
    candidates: list[CandidateScore]
    total_chunks_analyzed: int
    summary: str
