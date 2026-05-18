from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.score import (
    CandidateComparison,
    CandidateDiagnosis,
    CandidatePollTrend,
    CandidateScore,
    CandidateStrategy,
    ComparisonDimension,
    DailyVerdict,
    PollMethodAnalysis,
    PollTrendAnalysis,
    SearchResult,
)
from rag.verdict_graph import (
    _validate_consistency,
    _validate_grounding,
    _validate_probability,
    build_verdict_graph,
    run_verdict_graph,
    MAX_RETRIES,
    PROBABILITY_CHANGE_THRESHOLD,
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


def _make_verdict(
    candidates: list[dict] | None = None,
    summary: str = "판세 분석",
) -> DailyVerdict:
    if candidates is None:
        candidates = [
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.45},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "균형", "win_probability": 0.35},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.20},
        ]
    return DailyVerdict(
        district_id="pyeongtaek_b",
        district_name="평택을",
        date=datetime.now(),
        candidates=[
            CandidateScore(
                candidate=c["candidate"], party=c["party"], district_id="pyeongtaek_b",
                verdict=c["verdict"], win_probability=c["win_probability"],
                reasoning="테스트 근거", supporting_chunks=["c1"], chunk_count=1,
            )
            for c in candidates
        ],
        total_chunks_analyzed=2,
        summary=summary,
    )


class TestValidateGrounding:

    def test_pass_when_all_candidates_registered(self):
        verdict = _make_verdict()
        errors = _validate_grounding(verdict, SAMPLE_CHUNKS, DISTRICT)
        assert errors == []

    def test_fail_on_unregistered_candidate(self):
        verdict = _make_verdict([
            {"candidate": "홍길동", "party": "무소속", "verdict": "우세", "win_probability": 1.0},
        ])
        errors = _validate_grounding(verdict, SAMPLE_CHUNKS, DISTRICT)
        assert any("미등록 후보" in e for e in errors)

    def test_fail_on_no_evidence_but_winning(self):
        verdict = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.5},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "균형", "win_probability": 0.3},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "우세", "win_probability": 0.2},
        ])
        errors = _validate_grounding(verdict, SAMPLE_CHUNKS, DISTRICT)
        assert any("조국" in e and "근거 청크 0건" in e for e in errors)

    def test_pass_when_no_chunk_candidates(self):
        empty_chunks = [
            SearchResult(
                id="c1", score=0.9, text="일반 기사",
                article_url="https://a.com/1", source="naver", title="기사",
                published_at=datetime(2026, 4, 28), candidate="", district_id="pyeongtaek_b",
            ),
        ]
        verdict = _make_verdict()
        errors = _validate_grounding(verdict, empty_chunks, DISTRICT)
        assert errors == []


class TestValidateConsistency:

    def test_pass_when_no_previous(self):
        verdict = _make_verdict()
        errors = _validate_consistency(verdict, None)
        assert errors == []

    def test_pass_when_small_change(self):
        current = _make_verdict()
        previous = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.40},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "균형", "win_probability": 0.35},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.25},
        ])
        errors = _validate_consistency(current, previous)
        assert errors == []

    def test_fail_on_large_change(self):
        current = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.80},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "열세", "win_probability": 0.15},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.05},
        ])
        previous = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "균형", "win_probability": 0.35},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "우세", "win_probability": 0.40},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.25},
        ])
        errors = _validate_consistency(current, previous)
        assert any("급변" in e for e in errors)

    def test_threshold_boundary_pass(self):
        current = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.60},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "균형", "win_probability": 0.25},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.15},
        ])
        previous = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.45},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "균형", "win_probability": 0.35},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.20},
        ])
        errors = _validate_consistency(current, previous)
        assert errors == []


class TestValidateProbability:

    def test_pass_valid(self):
        verdict = _make_verdict()
        errors = _validate_probability(verdict)
        assert errors == []

    def test_fail_out_of_range(self):
        verdict = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 1.5},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "열세", "win_probability": -0.3},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": -0.2},
        ])
        errors = _validate_probability(verdict)
        assert len(errors) >= 2

    def test_fail_sum_not_one(self):
        verdict = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.5},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "균형", "win_probability": 0.5},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.5},
        ])
        errors = _validate_probability(verdict)
        assert any("합계" in e for e in errors)


class TestBuildVerdictGraph:

    def test_graph_compiles(self):
        graph = build_verdict_graph()
        assert graph is not None

    def test_graph_pass_on_valid_verdict(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(mock_scorer, SAMPLE_CHUNKS, DISTRICT)

        assert result is not None
        assert result.district_id == "pyeongtaek_b"
        assert len(result.candidates) == 3
        mock_scorer.score.assert_called_once()

    def test_graph_retries_on_validation_failure(self):
        bad_verdict = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.80},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "열세", "win_probability": 0.15},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.05},
        ])
        good_verdict = _make_verdict()

        previous = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "균형", "win_probability": 0.35},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "우세", "win_probability": 0.40},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.25},
        ])

        mock_scorer = MagicMock()
        mock_scorer.score.side_effect = [bad_verdict, good_verdict]

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = previous
            result = run_verdict_graph(mock_scorer, SAMPLE_CHUNKS, DISTRICT)

        assert mock_scorer.score.call_count == 2
        assert result is not None

    def test_graph_stops_at_max_retries(self):
        bad_verdict = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "우세", "win_probability": 0.90},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "열세", "win_probability": 0.08},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.02},
        ])

        previous = _make_verdict([
            {"candidate": "김용남", "party": "더불어민주당", "verdict": "균형", "win_probability": 0.35},
            {"candidate": "유의동", "party": "국민의힘", "verdict": "우세", "win_probability": 0.40},
            {"candidate": "조국", "party": "조국혁신당", "verdict": "열세", "win_probability": 0.25},
        ])

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = bad_verdict

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = previous
            result = run_verdict_graph(mock_scorer, SAMPLE_CHUNKS, DISTRICT)

        assert mock_scorer.score.call_count == MAX_RETRIES + 1
        assert result is not None

    def test_graph_no_previous_verdict(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(mock_scorer, SAMPLE_CHUNKS, DISTRICT)

        assert result is not None
        mock_scorer.score.assert_called_once()

    def test_graph_with_grouped_chunks(self):
        grouped = {
            "김용남": {"지지율": SAMPLE_CHUNKS[:1]},
            "유의동": {"지지율": SAMPLE_CHUNKS[1:]},
        }
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(mock_scorer, SAMPLE_CHUNKS, DISTRICT, grouped_chunks=grouped)

        assert result is not None
        call_kwargs = mock_scorer.score.call_args
        assert call_kwargs.kwargs.get("grouped_chunks") == grouped


SAMPLE_DIAGNOSIS = [
    CandidateDiagnosis(
        candidate="김용남", party="더불어민주당",
        problems=["인지도 부족", "조직력 약화"],
        root_causes=["늦은 출마 선언"],
        severity="주의", summary="전반적 인지도 부족",
    ),
]

SAMPLE_STRATEGY = [
    CandidateStrategy(
        candidate="김용남", party="더불어민주당",
        solutions=["SNS 홍보 강화", "지역 유세 확대"],
        action_plan=["단기: SNS 캠페인", "중기: 토론회 참석"],
        priority="긴급", expected_impact="인지도 10%p 상승 예상",
        summary="적극적 홍보 필요",
    ),
]

SAMPLE_COMPARISON = [
    CandidateComparison(
        candidate_a="김용남", candidate_b="유의동",
        dimensions=[
            ComparisonDimension(
                dimension="지지율",
                candidate_a_assessment="45% 선두",
                candidate_b_assessment="35% 추격",
                advantage="김용남 우위",
            ),
        ],
        overall_edge="김용남이 전반적 우위",
        summary="김용남이 지지율에서 앞서지만 격차 축소 중",
    ),
]


class TestAnalysisModes:

    def test_graph_compiles_diagnosis_mode(self):
        graph = build_verdict_graph(mode="diagnosis")
        assert graph is not None

    def test_graph_compiles_strategy_mode(self):
        graph = build_verdict_graph(mode="strategy")
        assert graph is not None

    def test_graph_compiles_comparison_mode(self):
        graph = build_verdict_graph(mode="comparison")
        assert graph is not None

    def test_diagnosis_mode_calls_diagnose(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()
        mock_scorer.diagnose.return_value = SAMPLE_DIAGNOSIS

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, SAMPLE_CHUNKS, DISTRICT, mode="diagnosis",
            )

        mock_scorer.score.assert_called_once()
        mock_scorer.diagnose.assert_called_once()
        assert result.diagnosis == SAMPLE_DIAGNOSIS
        assert result.analysis_mode == "diagnosis"

    def test_strategy_mode_calls_diagnose_and_strategize(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()
        mock_scorer.diagnose.return_value = SAMPLE_DIAGNOSIS
        mock_scorer.strategize.return_value = SAMPLE_STRATEGY

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, SAMPLE_CHUNKS, DISTRICT, mode="strategy",
            )

        mock_scorer.score.assert_called_once()
        mock_scorer.diagnose.assert_called_once()
        mock_scorer.strategize.assert_called_once()
        assert result.diagnosis == SAMPLE_DIAGNOSIS
        assert result.strategy == SAMPLE_STRATEGY
        assert result.analysis_mode == "strategy"

    def test_comparison_mode_calls_compare(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()
        mock_scorer.compare.return_value = SAMPLE_COMPARISON

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, SAMPLE_CHUNKS, DISTRICT,
                mode="comparison", target_candidates=["김용남", "유의동"],
            )

        mock_scorer.score.assert_called_once()
        mock_scorer.compare.assert_called_once()
        assert result.comparison == SAMPLE_COMPARISON
        assert result.analysis_mode == "comparison"

    def test_diagnosis_mode_with_target_candidates(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()
        mock_scorer.diagnose.return_value = SAMPLE_DIAGNOSIS

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, SAMPLE_CHUNKS, DISTRICT,
                mode="diagnosis", target_candidates=["김용남"],
            )

        call_kwargs = mock_scorer.diagnose.call_args
        assert call_kwargs.kwargs.get("target_candidates") == ["김용남"]

    def test_verdict_mode_has_no_extra_results(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, SAMPLE_CHUNKS, DISTRICT, mode="verdict",
            )

        assert result.diagnosis is None
        assert result.strategy is None
        assert result.comparison is None
        assert result.analysis_mode == "verdict"

    def test_comparison_defaults_to_first_two_candidates(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _make_verdict()
        mock_scorer.compare.return_value = SAMPLE_COMPARISON

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, SAMPLE_CHUNKS, DISTRICT, mode="comparison",
            )

        call_args = mock_scorer.compare.call_args
        assert call_args[0][3] == "김용남"
        assert call_args[0][4] == "유의동"

    def test_graph_compiles_opinion_polls_mode(self):
        graph = build_verdict_graph(mode="opinion_polls")
        assert graph is not None

    def test_opinion_polls_mode_calls_analyze_polls(self):
        from datetime import date
        from models.poll import PollEntry, PollMeta

        poll_entries = [
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
        poll_metas = [
            PollMeta(
                survey_date=date(2026, 5, 10), district_id="pyeongtaek_b",
                pollster="한국갤럽", method="무선전화면접",
                sample_size=1000, margin_of_error=3.1,
            ),
        ]

        poll_analysis = PollTrendAnalysis(
            total_surveys=1,
            analysis_period="2026-05-10 ~ 2026-05-10",
            candidate_trends=[
                CandidatePollTrend(
                    candidate="김용남", party="더불어민주당",
                    latest_support=38.5, trend_direction="정체",
                    trend_description="테스트",
                ),
            ],
            method_analysis=[
                PollMethodAnalysis(
                    method="무선전화면접", characteristics="테스트",
                    reliability_note="테스트", results_summary="테스트",
                ),
            ],
            key_findings=["발견 1"],
            trend_summary="테스트 요약",
            reliability_assessment="테스트 신뢰성",
        )

        verdict_with_polls = _make_verdict()
        verdict_with_polls.poll_analysis = poll_analysis
        verdict_with_polls.analysis_mode = "opinion_polls"

        mock_scorer = MagicMock()
        mock_scorer.analyze_polls.return_value = verdict_with_polls

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            result = run_verdict_graph(
                mock_scorer, [], DISTRICT,
                mode="opinion_polls",
                poll_entries=poll_entries,
                poll_metas=poll_metas,
            )

        mock_scorer.analyze_polls.assert_called_once()
        assert result.analysis_mode == "opinion_polls"
        assert result.poll_analysis is not None
        assert result.poll_analysis.total_surveys == 1
        assert len(result.poll_analysis.candidate_trends) == 1
        assert len(result.poll_analysis.method_analysis) == 1

    def test_opinion_polls_mode_no_score_called(self):
        from datetime import date
        from models.poll import PollEntry

        poll_entries = [
            PollEntry(
                district_id="pyeongtaek_b", candidate="김용남",
                party="더불어민주당", support=38.5, pollster="한국갤럽",
                survey_date=date(2026, 5, 10),
            ),
        ]

        mock_scorer = MagicMock()
        mock_scorer.analyze_polls.return_value = _make_verdict()

        with patch("rag.verdict_graph.VerdictStore") as MockStore:
            MockStore.return_value.load_latest.return_value = None
            run_verdict_graph(
                mock_scorer, [], DISTRICT,
                mode="opinion_polls",
                poll_entries=poll_entries,
            )

        mock_scorer.score.assert_not_called()
        mock_scorer.analyze_polls.assert_called_once()
