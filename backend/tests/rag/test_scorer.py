from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.score import CandidateScore, DailyVerdict, PollTrendAnalysis, SearchResult
from rag.scorer import (
    AbstractScorer,
    ScorerRegistry,
    _build_opinion_polls_user_prompt,
    _build_user_prompt,
    _normalize_probabilities,
    _parse_llm_response,
    _parse_opinion_polls_response,
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


VALID_OPINION_POLLS_RESPONSE = json.dumps({
    "candidates": [
        {"candidate": "김용남", "verdict": "균형", "win_probability": 0.40,
         "reasoning": "여론조사 수치상 근소 우위이나 오차범위 이내"},
        {"candidate": "유의동", "verdict": "균형", "win_probability": 0.35,
         "reasoning": "접전 양상, 상승 추세"},
        {"candidate": "조국", "verdict": "열세", "win_probability": 0.25,
         "reasoning": "지지율 하위권"},
    ],
    "poll_analysis": {
        "total_surveys": 2,
        "analysis_period": "2026-05-01 ~ 2026-05-10",
        "candidate_trends": [
            {"candidate": "김용남", "party": "더불어민주당",
             "latest_support": 38.5, "trend_direction": "정체",
             "trend_description": "최근 2회 조사에서 38~39% 수준 유지"},
            {"candidate": "유의동", "party": "국민의힘",
             "latest_support": 35.2, "trend_direction": "상승",
             "trend_description": "32%에서 35%로 3%p 상승"},
        ],
        "method_analysis": [
            {"method": "무선전화면접",
             "characteristics": "조사원 직접 면접으로 높은 신뢰도",
             "reliability_note": "진보 정당 약간 높게 나오는 경향",
             "results_summary": "김용남 40.1%, 유의동 34.2%"},
            {"method": "무선전화ARS",
             "characteristics": "자동응답으로 대규모 표본 확보",
             "reliability_note": "보수 정당 높게 나오는 경향",
             "results_summary": "김용남 36.9%, 유의동 36.2%"},
        ],
        "key_findings": [
            "면접 조사와 ARS 조사 간 약 3%p 차이 관찰",
            "유의동 후보의 상승 추세가 뚜렷",
        ],
        "trend_summary": "전반적으로 접전 양상이며 격차 축소 중",
        "reliability_assessment": "2회 조사로 추세 판단에 한계, 추가 조사 필요",
    },
    "summary": "접전 양상의 여론조사 동향",
}, ensure_ascii=False)


class TestBuildOpinionPollsUserPrompt:

    def _make_poll_entries(self):
        from datetime import date
        from models.poll import PollEntry
        return [
            PollEntry(
                district_id="pyeongtaek_b", candidate="김용남",
                party="더불어민주당", support=38.5, pollster="한국갤럽",
                survey_date=date(2026, 5, 10),
            ),
            PollEntry(
                district_id="pyeongtaek_b", candidate="유의동",
                party="국민의힘", support=35.2, pollster="한국갤럽",
                survey_date=date(2026, 5, 10),
            ),
        ]

    def _make_poll_metas(self):
        from datetime import date
        from models.poll import PollMeta
        return [
            PollMeta(
                survey_date=date(2026, 5, 10), district_id="pyeongtaek_b",
                pollster="한국갤럽", method="무선전화면접",
                sample_size=1000, margin_of_error=3.1,
                source_url="https://example.com/poll",
            ),
        ]

    def test_contains_district_name(self):
        prompt = _build_opinion_polls_user_prompt(
            self._make_poll_entries(), [], DISTRICT,
        )
        assert "평택을" in prompt

    def test_contains_candidates(self):
        prompt = _build_opinion_polls_user_prompt(
            self._make_poll_entries(), [], DISTRICT,
        )
        assert "김용남" in prompt
        assert "유의동" in prompt

    def test_contains_support_rates(self):
        prompt = _build_opinion_polls_user_prompt(
            self._make_poll_entries(), [], DISTRICT,
        )
        assert "38.5%" in prompt
        assert "35.2%" in prompt

    def test_contains_meta_info(self):
        prompt = _build_opinion_polls_user_prompt(
            self._make_poll_entries(), self._make_poll_metas(), DISTRICT,
        )
        assert "무선전화면접" in prompt
        assert "1000" in prompt
        assert "3.1" in prompt

    def test_contains_source_url(self):
        prompt = _build_opinion_polls_user_prompt(
            self._make_poll_entries(), self._make_poll_metas(), DISTRICT,
        )
        assert "https://example.com/poll" in prompt

    def test_empty_polls(self):
        prompt = _build_opinion_polls_user_prompt([], [], DISTRICT)
        assert "0회 조사" in prompt

    def test_includes_optional_chunks(self):
        prompt = _build_opinion_polls_user_prompt(
            self._make_poll_entries(), [], DISTRICT, chunks=SAMPLE_CHUNKS,
        )
        assert "여론조사 관련 뉴스 기사" in prompt


class TestParseOpinionPollsResponse:

    def test_valid_response(self):
        verdict = _parse_opinion_polls_response(VALID_OPINION_POLLS_RESPONSE, DISTRICT)

        assert isinstance(verdict, DailyVerdict)
        assert verdict.district_id == "pyeongtaek_b"
        assert verdict.analysis_mode == "opinion_polls"
        assert len(verdict.candidates) == 3
        assert verdict.poll_analysis is not None

    def test_poll_analysis_fields(self):
        verdict = _parse_opinion_polls_response(VALID_OPINION_POLLS_RESPONSE, DISTRICT)
        pa = verdict.poll_analysis

        assert pa.total_surveys == 2
        assert pa.analysis_period == "2026-05-01 ~ 2026-05-10"
        assert len(pa.candidate_trends) == 2
        assert len(pa.method_analysis) == 2
        assert len(pa.key_findings) == 2

    def test_candidate_trend_fields(self):
        verdict = _parse_opinion_polls_response(VALID_OPINION_POLLS_RESPONSE, DISTRICT)
        ct = verdict.poll_analysis.candidate_trends[0]

        assert ct.candidate == "김용남"
        assert ct.latest_support == 38.5
        assert ct.trend_direction == "정체"

    def test_method_analysis_fields(self):
        verdict = _parse_opinion_polls_response(VALID_OPINION_POLLS_RESPONSE, DISTRICT)
        ma = verdict.poll_analysis.method_analysis[0]

        assert ma.method == "무선전화면접"
        assert "신뢰도" in ma.characteristics

    def test_probabilities_normalized(self):
        verdict = _parse_opinion_polls_response(VALID_OPINION_POLLS_RESPONSE, DISTRICT)
        total = sum(c.win_probability for c in verdict.candidates)
        assert abs(total - 1.0) < 0.01

    def test_markdown_wrapped(self):
        wrapped = f"```json\n{VALID_OPINION_POLLS_RESPONSE}\n```"
        verdict = _parse_opinion_polls_response(wrapped, DISTRICT)
        assert verdict.poll_analysis is not None


class TestAnalyzePolls:

    def test_analyze_polls_calls_llm(self):
        from datetime import date
        from models.poll import PollEntry

        class FakeScorer(AbstractScorer):
            @property
            def name(self):
                return "fake"

            def _call_llm(self, system, user, **kwargs):
                return VALID_OPINION_POLLS_RESPONSE

        entries = [
            PollEntry(
                district_id="pyeongtaek_b", candidate="김용남",
                party="더불어민주당", support=38.5, pollster="한국갤럽",
                survey_date=date(2026, 5, 10),
            ),
        ]

        scorer = FakeScorer()
        verdict = scorer.analyze_polls(entries, [], DISTRICT)

        assert isinstance(verdict, DailyVerdict)
        assert verdict.poll_analysis is not None
        assert verdict.analysis_mode == "opinion_polls"

    def test_analyze_polls_empty_returns_equal_probability(self):
        class FakeScorer(AbstractScorer):
            @property
            def name(self):
                return "fake"

            def _call_llm(self, system, user, **kwargs):
                raise AssertionError("should not call LLM")

        scorer = FakeScorer()
        verdict = scorer.analyze_polls([], [], DISTRICT)

        assert len(verdict.candidates) == 3
        assert all(c.verdict == "균형" for c in verdict.candidates)

    def test_analyze_polls_parse_failure_returns_equal(self):
        from datetime import date
        from models.poll import PollEntry

        class BadScorer(AbstractScorer):
            @property
            def name(self):
                return "bad"

            def _call_llm(self, system, user, **kwargs):
                return "invalid json"

        entries = [
            PollEntry(
                district_id="pyeongtaek_b", candidate="김용남",
                party="더불어민주당", support=38.5, pollster="한국갤럽",
                survey_date=date(2026, 5, 10),
            ),
        ]

        scorer = BadScorer()
        verdict = scorer.analyze_polls(entries, [], DISTRICT)
        assert all(c.verdict == "균형" for c in verdict.candidates)


class TestScorerRegistry:

    def test_openai_registered(self):
        import rag.openai_scorer  # noqa: F401
        assert "openai" in ScorerRegistry.registered_names

    def test_anthropic_registered(self):
        import rag.anthropic_scorer  # noqa: F401
        assert "anthropic" in ScorerRegistry.registered_names
