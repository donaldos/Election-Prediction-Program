from __future__ import annotations

from datetime import datetime
from typing import Union

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


class CandidateReasoning(BaseModel):
    """후보별 판정 근거 상세."""

    support_rate: str
    pledge_reaction: str
    strengths: str
    weaknesses: str
    issues: str
    support_trend: str
    public_opinion: str
    strategy: str
    forecast: str


class CandidateDiagnosis(BaseModel):
    """후보별 문제점 진단."""

    candidate: str
    party: str
    problems: list[str]
    root_causes: list[str]
    severity: str
    summary: str


class CandidateStrategy(BaseModel):
    """후보별 대응방안/전략."""

    candidate: str
    party: str
    solutions: list[str]
    action_plan: list[str]
    priority: str
    expected_impact: str
    summary: str


class ComparisonDimension(BaseModel):
    """비교 분석 항목."""

    dimension: str
    candidate_a_assessment: str
    candidate_b_assessment: str
    advantage: str


class CandidateComparison(BaseModel):
    """후보 간 비교 분석."""

    candidate_a: str
    candidate_b: str
    dimensions: list[ComparisonDimension]
    overall_edge: str
    summary: str


class CandidateScore(BaseModel):
    """후보별 판정 결과. Scorer 출력."""

    candidate: str
    party: str
    district_id: str

    verdict: str
    win_probability: float
    reasoning: Union[str, CandidateReasoning]

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
    analysis_mode: str = "verdict"
    diagnosis: list[CandidateDiagnosis] | None = None
    strategy: list[CandidateStrategy] | None = None
    comparison: list[CandidateComparison] | None = None
