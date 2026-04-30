from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.score import CandidateScore, DailyVerdict, SearchResult
from rag.scorer import (
    AbstractScorer,
    ScorerRegistry,
    _build_user_prompt,
    _normalize_probabilities,
    _parse_llm_response,
)


DISTRICT = {
    "id": "pyeongtaek_b",
    "name": "평택을",
    "candidates": [
        {"name": "김용남", "party": "더불어민주당"},
        {"name": "유의동", "party": "국민의힘"},
        {"name": "조국", "party": "조국혁신당"},
    ],
}

SAMPLE_CHUNKS = [
    SearchResult(
        id="c1", score=0.9, text="김용남 후보가 여론조사에서 선두",
        article_url="https://a.com/1", source="naver", title="여론조사",
        published_at=datetime(2026, 4, 28), candidate="김용남", district_id="pyeongtaek_b",
    ),
    SearchResult(
        id="c2", score=0.8, text="유의동 후보가 추격 중",
        article_url="https://a.com/2", source="naver", title="접전",
        published_at=datetime(2026, 4, 29), candidate="유의동", district_id="pyeongtaek_b",
    ),
]

VALID_LLM_RESPONSE = json.dumps({
    "candidates": [
        {"candidate": "김용남", "verdict": "우세", "win_probability": 0.45, "reasoning": "여론조사 선두"},
        {"candidate": "유의동", "verdict": "균형", "win_probability": 0.35, "reasoning": "추격 중"},
        {"candidate": "조국", "verdict": "열세", "win_probability": 0.20, "reasoning": "지지율 낮음"},
    ],
    "summary": "김용남이 근소하게 앞서는 양상",
}, ensure_ascii=False)


class TestNormalizeProbabilities:

    def test_already_normalized(self):
        scores = [
            CandidateScore(candidate="A", party="P", district_id="d", verdict="우세",
                           win_probability=0.6, reasoning="r", supporting_chunks=[], chunk_count=0),
            CandidateScore(candidate="B", party="P", district_id="d", verdict="열세",
                           win_probability=0.4, reasoning="r", supporting_chunks=[], chunk_count=0),
        ]
        result = _normalize_probabilities(scores)
        assert abs(sum(s.win_probability for s in result) - 1.0) < 0.01

    def test_normalizes_when_sum_not_one(self):
        scores = [
            CandidateScore(candidate="A", party="P", district_id="d", verdict="우세",
                           win_probability=0.6, reasoning="r", supporting_chunks=[], chunk_count=0),
            CandidateScore(candidate="B", party="P", district_id="d", verdict="열세",
                           win_probability=0.6, reasoning="r", supporting_chunks=[], chunk_count=0),
        ]
        result = _normalize_probabilities(scores)
        assert abs(sum(s.win_probability for s in result) - 1.0) < 0.01

    def test_all_zero_gives_equal(self):
        scores = [
            CandidateScore(candidate="A", party="P", district_id="d", verdict="균형",
                           win_probability=0.0, reasoning="r", supporting_chunks=[], chunk_count=0),
            CandidateScore(candidate="B", party="P", district_id="d", verdict="균형",
                           win_probability=0.0, reasoning="r", supporting_chunks=[], chunk_count=0),
        ]
        result = _normalize_probabilities(scores)
        assert all(s.win_probability == 0.5 for s in result)


class TestParseLlmResponse:

    def test_valid_response(self):
        verdict = _parse_llm_response(VALID_LLM_RESPONSE, DISTRICT, SAMPLE_CHUNKS)

        assert isinstance(verdict, DailyVerdict)
        assert verdict.district_id == "pyeongtaek_b"
        assert len(verdict.candidates) == 3
        assert verdict.summary == "김용남이 근소하게 앞서는 양상"

        kim = next(c for c in verdict.candidates if c.candidate == "김용남")
        assert kim.verdict == "우세"
        assert kim.party == "더불어민주당"

    def test_probabilities_sum_to_one(self):
        verdict = _parse_llm_response(VALID_LLM_RESPONSE, DISTRICT, SAMPLE_CHUNKS)
        total = sum(c.win_probability for c in verdict.candidates)
        assert abs(total - 1.0) < 0.01

    def test_markdown_wrapped_json(self):
        wrapped = f"```json\n{VALID_LLM_RESPONSE}\n```"
        verdict = _parse_llm_response(wrapped, DISTRICT, SAMPLE_CHUNKS)
        assert len(verdict.candidates) == 3

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("이것은 JSON이 아닙니다", DISTRICT, SAMPLE_CHUNKS)

    def test_verdict_values(self):
        verdict = _parse_llm_response(VALID_LLM_RESPONSE, DISTRICT, SAMPLE_CHUNKS)
        valid_verdicts = {"우세", "균형", "열세"}
        for c in verdict.candidates:
            assert c.verdict in valid_verdicts


class TestBuildUserPrompt:

    def test_contains_district_name(self):
        prompt = _build_user_prompt(SAMPLE_CHUNKS, DISTRICT)
        assert "평택을" in prompt

    def test_contains_all_candidates(self):
        prompt = _build_user_prompt(SAMPLE_CHUNKS, DISTRICT)
        for cand in DISTRICT["candidates"]:
            assert cand["name"] in prompt

    def test_contains_chunk_text(self):
        prompt = _build_user_prompt(SAMPLE_CHUNKS, DISTRICT)
        assert "김용남 후보가 여론조사에서 선두" in prompt

    def test_empty_chunks(self):
        prompt = _build_user_prompt([], DISTRICT)
        assert "0건" in prompt


class TestAbstractScorer:

    def test_score_calls_llm(self):
        class FakeScorer(AbstractScorer):
            @property
            def name(self):
                return "fake"

            def _call_llm(self, system, user):
                return VALID_LLM_RESPONSE

        scorer = FakeScorer()
        verdict = scorer.score(SAMPLE_CHUNKS, DISTRICT)

        assert isinstance(verdict, DailyVerdict)
        assert len(verdict.candidates) == 3
        assert verdict.total_chunks_analyzed == 2

    def test_empty_chunks_returns_equal_probability(self):
        class FakeScorer(AbstractScorer):
            @property
            def name(self):
                return "fake"

            def _call_llm(self, system, user):
                raise AssertionError("should not call LLM")

        scorer = FakeScorer()
        verdict = scorer.score([], DISTRICT)

        assert len(verdict.candidates) == 3
        assert all(c.verdict == "균형" for c in verdict.candidates)
        total = sum(c.win_probability for c in verdict.candidates)
        assert abs(total - 1.0) < 0.01

    def test_parse_failure_returns_equal_probability(self):
        class BadScorer(AbstractScorer):
            @property
            def name(self):
                return "bad"

            def _call_llm(self, system, user):
                return "invalid json response"

        scorer = BadScorer()
        verdict = scorer.score(SAMPLE_CHUNKS, DISTRICT)

        assert all(c.verdict == "균형" for c in verdict.candidates)


class TestScorerRegistry:

    def test_openai_registered(self):
        import rag.openai_scorer  # noqa: F401
        assert "openai" in ScorerRegistry.registered_names

    def test_anthropic_registered(self):
        import rag.anthropic_scorer  # noqa: F401
        assert "anthropic" in ScorerRegistry.registered_names
