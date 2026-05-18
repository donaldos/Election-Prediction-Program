"""LangGraph 기반 RAG 멀티스테이지 오케스트레이션.

분석 모드별 그래프 플로우:
    verdict:    score → validate → END
    diagnosis:  score → validate → diagnose → END
    strategy:   score → validate → diagnose → strategize → END
    comparison: score → validate → compare → END

공통 흐름: validate 실패 시 correct → score 재시도 (최대 MAX_RETRIES 회)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from models.score import (
    CandidateComparison,
    CandidateDiagnosis,
    CandidateScore,
    CandidateStrategy,
    DailyVerdict,
    PollTrendAnalysis,
    SearchResult,
)
from rag.graph_logger import GraphLogger
from rag.scorer import AbstractScorer, flatten_grouped_chunks
from rag.verdict_store import VerdictStore

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
PROBABILITY_CHANGE_THRESHOLD = 0.30

AnalysisMode = Literal["verdict", "diagnosis", "strategy", "comparison", "opinion_polls"]


class VerdictState(TypedDict):
    """그래프 상태."""

    chunks: list[SearchResult]
    grouped_chunks: dict[str, dict[str, list[SearchResult]]] | None
    district: dict
    scorer: AbstractScorer
    verdict: DailyVerdict | None
    previous_verdict: DailyVerdict | None
    validation_errors: list[str]
    retry_count: int
    graph_logger: GraphLogger
    analysis_mode: str
    target_candidates: list[str] | None
    diagnosis: list[CandidateDiagnosis] | None
    strategy: list[CandidateStrategy] | None
    comparison: list[CandidateComparison] | None
    poll_entries: list | None
    poll_metas: list | None
    poll_analysis: PollTrendAnalysis | None


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def _candidates_summary(verdict: DailyVerdict) -> list[dict]:
    return [
        {
            "candidate": s.candidate,
            "verdict": s.verdict,
            "win_probability": round(s.win_probability, 4),
        }
        for s in verdict.candidates
    ]


# ---------------------------------------------------------------------------
# 노드: score / validate / correct (기존)
# ---------------------------------------------------------------------------

def score_node(state: VerdictState) -> dict:
    """LLM 판정 노드 — Scorer.score() 호출."""
    scorer = state["scorer"]
    chunks = state["chunks"]
    district = state["district"]
    grouped = state["grouped_chunks"]
    retry = state["retry_count"]
    gl = state["graph_logger"]

    logger.info(
        "[score 노드] 진입 — %s, 청크 %d건, retry=%d/%d",
        district["name"], len(chunks), retry, MAX_RETRIES,
    )
    gl.log_score_enter(district["name"], len(chunks), retry)

    verdict = scorer.score(chunks, district, grouped_chunks=grouped)

    candidates = _candidates_summary(verdict)
    logger.info(
        "[score 노드] 완료 — %s",
        ", ".join(
            f"{s.candidate}={s.verdict}({s.win_probability:.1%})"
            for s in verdict.candidates
        ),
    )
    gl.log_score_exit(candidates)

    return {"verdict": verdict}


def validate_node(state: VerdictState) -> dict:
    """검증 노드 — 근거 일치 + 일관성 검사."""
    verdict = state["verdict"]
    chunks = state["chunks"]
    district = state["district"]
    previous = state["previous_verdict"]
    gl = state["graph_logger"]
    errors: list[str] = []

    has_previous = previous is not None
    logger.info(
        "[validate 노드] 진입 — %s, 직전 판정 %s",
        district["name"], "있음" if has_previous else "없음",
    )
    gl.log_validate_enter(has_previous)

    if not verdict or verdict.total_chunks_analyzed == 0:
        logger.info("[validate 노드] 판정 데이터 없음 — 검증 생략")
        gl.log_validate_exit("skip", "skip", "skip", [], [], [])
        return {"validation_errors": errors}

    grounding_errors = _validate_grounding(verdict, chunks, district)
    consistency_errors = _validate_consistency(verdict, previous)
    probability_errors = _validate_probability(verdict)

    g_status = "fail" if grounding_errors else "pass"
    c_status = "fail" if consistency_errors else "pass"
    p_status = "fail" if probability_errors else "pass"

    if grounding_errors:
        logger.warning("[validate 노드] 근거 검증 실패 — %s", "; ".join(grounding_errors))
    else:
        logger.info("[validate 노드] 근거 검증 통과")

    if consistency_errors:
        logger.warning("[validate 노드] 일관성 검증 실패 — %s", "; ".join(consistency_errors))
    else:
        logger.info("[validate 노드] 일관성 검증 통과")

    if probability_errors:
        logger.warning("[validate 노드] 확률 검증 실패 — %s", "; ".join(probability_errors))
    else:
        logger.info("[validate 노드] 확률 검증 통과")

    errors = grounding_errors + consistency_errors + probability_errors

    logger.info(
        "[validate 노드] 완료 — 총 %d건 오류 (근거 %d, 일관성 %d, 확률 %d)",
        len(errors), len(grounding_errors), len(consistency_errors), len(probability_errors),
    )
    gl.log_validate_exit(
        g_status, c_status, p_status,
        grounding_errors, consistency_errors, probability_errors,
    )

    return {"validation_errors": errors}


def correct_node(state: VerdictState) -> dict:
    """보정 노드 — 검증 실패 시 재판정 요청."""
    errors = state["validation_errors"]
    retry = state["retry_count"]
    district = state["district"]
    gl = state["graph_logger"]

    logger.info(
        "[correct 노드] 진입 — %s, 검증 오류 %d건, 재시도 %d→%d",
        district["name"], len(errors), retry, retry + 1,
    )
    for i, err in enumerate(errors, 1):
        logger.info("[correct 노드]   오류 %d: %s", i, err)

    gl.log_correct_enter(errors, retry, retry + 1)

    return {
        "retry_count": retry + 1,
        "validation_errors": [],
        "verdict": None,
    }


def route_validation(state: VerdictState) -> str:
    """검증 결과에 따른 분기."""
    errors = state["validation_errors"]
    retry = state["retry_count"]
    district = state["district"]
    gl = state["graph_logger"]

    if not errors:
        logger.info("[route] %s — 검증 통과 → 다음 단계", district["name"])
        gl.log_route("pass", "검증 통과")
        return "pass"

    if retry >= MAX_RETRIES:
        reason = f"최대 재시도 도달 ({MAX_RETRIES}회), 오류 {len(errors)}건 잔존"
        logger.warning(
            "[route] %s — %s → 다음 단계 (현재 판정 유지)",
            district["name"], reason,
        )
        gl.log_route("max_retry", reason)
        return "max_retry"

    reason = f"검증 실패 ({len(errors)}건), retry {retry}/{MAX_RETRIES}"
    logger.info(
        "[route] %s — %s → correct 노드로 분기",
        district["name"], reason,
    )
    gl.log_route("fail", reason)
    return "fail"


# ---------------------------------------------------------------------------
# 노드: diagnose / strategize / compare (신규)
# ---------------------------------------------------------------------------

def diagnose_node(state: VerdictState) -> dict:
    """문제점 진단 노드 — Scorer.diagnose() 호출."""
    scorer = state["scorer"]
    verdict = state["verdict"]
    chunks = state["chunks"]
    district = state["district"]
    targets = state.get("target_candidates")
    gl = state["graph_logger"]

    target_desc = ", ".join(targets) if targets else "전체"
    logger.info("[diagnose 노드] 진입 — %s, 대상: %s", district["name"], target_desc)
    gl.log_analysis_enter("diagnose", target_desc)

    diagnosis = scorer.diagnose(verdict, chunks, district, target_candidates=targets)

    logger.info(
        "[diagnose 노드] 완료 — %d명 진단",
        len(diagnosis),
    )
    gl.log_analysis_exit("diagnose", len(diagnosis))

    return {"diagnosis": diagnosis}


def strategize_node(state: VerdictState) -> dict:
    """전략 도출 노드 — Scorer.strategize() 호출."""
    scorer = state["scorer"]
    verdict = state["verdict"]
    diagnosis = state.get("diagnosis") or []
    chunks = state["chunks"]
    district = state["district"]
    targets = state.get("target_candidates")
    gl = state["graph_logger"]

    target_desc = ", ".join(targets) if targets else "전체"
    logger.info("[strategize 노드] 진입 — %s, 대상: %s", district["name"], target_desc)
    gl.log_analysis_enter("strategize", target_desc)

    strategies = scorer.strategize(
        verdict, diagnosis, chunks, district, target_candidates=targets,
    )

    logger.info(
        "[strategize 노드] 완료 — %d명 전략 도출",
        len(strategies),
    )
    gl.log_analysis_exit("strategize", len(strategies))

    return {"strategy": strategies}


def compare_node(state: VerdictState) -> dict:
    """후보 비교 노드 — Scorer.compare() 호출."""
    scorer = state["scorer"]
    verdict = state["verdict"]
    chunks = state["chunks"]
    district = state["district"]
    targets = state.get("target_candidates") or []
    gl = state["graph_logger"]

    if len(targets) < 2:
        candidates = [c["name"] for c in district.get("candidates", [])]
        targets = candidates[:2]

    logger.info("[compare 노드] 진입 — %s vs %s", targets[0], targets[1])
    gl.log_analysis_enter("compare", f"{targets[0]} vs {targets[1]}")

    comparisons = scorer.compare(
        verdict, chunks, district, targets[0], targets[1],
    )

    logger.info("[compare 노드] 완료 — %d건 비교", len(comparisons))
    gl.log_analysis_exit("compare", len(comparisons))

    return {"comparison": comparisons}


def analyze_polls_node(state: VerdictState) -> dict:
    """여론조사 분석 노드 — Scorer.analyze_polls() 호출."""
    scorer = state["scorer"]
    district = state["district"]
    chunks = state["chunks"]
    poll_entries = state.get("poll_entries") or []
    poll_metas = state.get("poll_metas") or []
    gl = state["graph_logger"]

    logger.info(
        "[analyze_polls 노드] 진입 — %s, 조사 %d건, 메타 %d건",
        district["name"], len(poll_entries), len(poll_metas),
    )
    gl.log_analysis_enter("analyze_polls", f"조사 {len(poll_entries)}건")

    verdict = scorer.analyze_polls(
        poll_entries, poll_metas, district,
        chunks=chunks if chunks else None,
    )

    logger.info(
        "[analyze_polls 노드] 완료 — %s",
        ", ".join(
            f"{s.candidate}={s.verdict}({s.win_probability:.1%})"
            for s in verdict.candidates
        ),
    )
    gl.log_analysis_exit("analyze_polls", len(verdict.candidates))

    return {
        "verdict": verdict,
        "poll_analysis": verdict.poll_analysis,
    }


# ---------------------------------------------------------------------------
# 검증 함수
# ---------------------------------------------------------------------------

def _validate_grounding(
    verdict: DailyVerdict,
    chunks: list[SearchResult],
    district: dict,
) -> list[str]:
    """판정 결과가 근거 청크와 일치하는지 검증."""
    errors: list[str] = []

    candidate_names = {c["name"] for c in district.get("candidates", [])}
    chunk_candidates = {c.candidate for c in chunks if c.candidate}

    for score in verdict.candidates:
        if score.candidate not in candidate_names:
            errors.append(f"미등록 후보 판정: {score.candidate}")
            continue

        if chunk_candidates and score.candidate not in chunk_candidates:
            if score.verdict == "우세":
                errors.append(
                    f"{score.candidate}: 근거 청크 0건인데 '우세' 판정"
                )

    return errors


def _validate_consistency(
    verdict: DailyVerdict,
    previous: DailyVerdict | None,
) -> list[str]:
    """직전 판정 대비 급격한 변동 감지."""
    if previous is None:
        return []

    errors: list[str] = []
    prev_map = {s.candidate: s.win_probability for s in previous.candidates}

    for score in verdict.candidates:
        prev_prob = prev_map.get(score.candidate)
        if prev_prob is None:
            continue

        delta = abs(score.win_probability - prev_prob)
        if delta > PROBABILITY_CHANGE_THRESHOLD:
            errors.append(
                f"{score.candidate}: 승률 급변 {prev_prob:.1%} → {score.win_probability:.1%} "
                f"(변동 {delta:.1%} > 임계값 {PROBABILITY_CHANGE_THRESHOLD:.0%})"
            )

    return errors


def _validate_probability(verdict: DailyVerdict) -> list[str]:
    """확률 합계 및 범위 검증."""
    errors: list[str] = []

    for score in verdict.candidates:
        if score.win_probability < 0 or score.win_probability > 1:
            errors.append(f"{score.candidate}: 확률 범위 초과 ({score.win_probability})")

    total = sum(s.win_probability for s in verdict.candidates)
    if abs(total - 1.0) > 0.05:
        errors.append(f"확률 합계 {total:.4f} ≠ 1.0")

    return errors


# ---------------------------------------------------------------------------
# 그래프 빌드 팩토리
# ---------------------------------------------------------------------------

_GRAPH_DESCRIPTIONS = {
    "verdict":       "score → validate → END",
    "diagnosis":     "score → validate → diagnose → END",
    "strategy":      "score → validate → diagnose → strategize → END",
    "comparison":    "score → validate → compare → END",
    "opinion_polls": "analyze_polls → END",
}


def build_verdict_graph(mode: AnalysisMode = "verdict") -> StateGraph:
    """분석 모드에 따른 그래프 빌드."""
    graph = StateGraph(VerdictState)

    graph.add_node("score", score_node)
    graph.add_node("validate", validate_node)
    graph.add_node("correct", correct_node)

    graph.set_entry_point("score")
    graph.add_edge("score", "validate")

    if mode == "verdict":
        graph.add_conditional_edges("validate", route_validation, {
            "pass": END,
            "fail": "correct",
            "max_retry": END,
        })
    elif mode == "diagnosis":
        graph.add_node("diagnose", diagnose_node)
        graph.add_conditional_edges("validate", route_validation, {
            "pass": "diagnose",
            "fail": "correct",
            "max_retry": "diagnose",
        })
        graph.add_edge("diagnose", END)
    elif mode == "strategy":
        graph.add_node("diagnose", diagnose_node)
        graph.add_node("strategize", strategize_node)
        graph.add_conditional_edges("validate", route_validation, {
            "pass": "diagnose",
            "fail": "correct",
            "max_retry": "diagnose",
        })
        graph.add_edge("diagnose", "strategize")
        graph.add_edge("strategize", END)
    elif mode == "comparison":
        graph.add_node("compare", compare_node)
        graph.add_conditional_edges("validate", route_validation, {
            "pass": "compare",
            "fail": "correct",
            "max_retry": "compare",
        })
        graph.add_edge("compare", END)
    elif mode == "opinion_polls":
        graph = StateGraph(VerdictState)
        graph.add_node("analyze_polls", analyze_polls_node)
        graph.set_entry_point("analyze_polls")
        graph.add_edge("analyze_polls", END)
        return graph.compile()

    graph.add_edge("correct", "score")

    return graph.compile()


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------

def run_verdict_graph(
    scorer: AbstractScorer,
    chunks: list[SearchResult],
    district: dict,
    grouped_chunks: dict[str, dict[str, list[SearchResult]]] | None = None,
    *,
    mode: AnalysisMode = "verdict",
    target_candidates: list[str] | None = None,
    poll_entries: list | None = None,
    poll_metas: list | None = None,
) -> DailyVerdict:
    """LangGraph 멀티스테이지 오케스트레이션 실행."""
    store = VerdictStore()
    previous = store.load_latest(district["id"])
    gl = GraphLogger(district["id"])

    has_previous = previous is not None
    if has_previous:
        logger.info(
            "직전 판정 로드 — %s (%s)",
            district["name"], previous.date.strftime("%Y-%m-%d %H:%M"),
        )

    graph_desc = _GRAPH_DESCRIPTIONS.get(mode, mode)

    initial_state: VerdictState = {
        "chunks": chunks,
        "grouped_chunks": grouped_chunks,
        "district": district,
        "scorer": scorer,
        "verdict": None,
        "previous_verdict": previous,
        "validation_errors": [],
        "retry_count": 0,
        "graph_logger": gl,
        "analysis_mode": mode,
        "target_candidates": target_candidates,
        "diagnosis": None,
        "strategy": None,
        "comparison": None,
        "poll_entries": poll_entries,
        "poll_metas": poll_metas,
        "poll_analysis": None,
    }

    logger.info("=" * 60)
    logger.info("LangGraph 시작 — %s [%s 모드]", district["name"], mode)
    logger.info("  그래프: %s", graph_desc)
    logger.info("  최대 재시도: %d회, 일관성 임계값: %.0f%%", MAX_RETRIES, PROBABILITY_CHANGE_THRESHOLD * 100)
    if target_candidates:
        logger.info("  대상 후보: %s", ", ".join(target_candidates))
    logger.info("=" * 60)

    gl.log_graph_start(
        district_name=district["name"],
        chunk_count=len(chunks),
        max_retries=MAX_RETRIES,
        threshold=PROBABILITY_CHANGE_THRESHOLD,
        has_previous=has_previous,
        analysis_mode=mode,
        target_candidates=target_candidates,
    )

    graph = build_verdict_graph(mode)
    final_state = graph.invoke(initial_state)

    verdict = final_state["verdict"]
    retry_count = final_state["retry_count"]
    errors = final_state["validation_errors"]

    verdict.analysis_mode = mode
    verdict.diagnosis = final_state.get("diagnosis")
    verdict.strategy = final_state.get("strategy")
    verdict.comparison = final_state.get("comparison")
    verdict.poll_analysis = final_state.get("poll_analysis")

    logger.info("=" * 60)
    if retry_count > 0 and errors:
        logger.warning(
            "LangGraph 완료 — %s [%s], 재시도 %d회, 미해소 오류 %d건",
            district["name"], mode, retry_count, len(errors),
        )
    elif retry_count > 0:
        logger.info(
            "LangGraph 완료 — %s [%s], 재시도 %d회 후 검증 통과",
            district["name"], mode, retry_count,
        )
    else:
        logger.info("LangGraph 완료 — %s [%s], 1회 판정으로 검증 통과", district["name"], mode)

    logger.info(
        "  판정: %s",
        ", ".join(
            f"{s.candidate}={s.verdict}({s.win_probability:.1%})"
            for s in verdict.candidates
        ),
    )
    if verdict.diagnosis:
        logger.info("  진단: %d명", len(verdict.diagnosis))
    if verdict.strategy:
        logger.info("  전략: %d명", len(verdict.strategy))
    if verdict.comparison:
        logger.info("  비교: %d건", len(verdict.comparison))
    if verdict.poll_analysis:
        logger.info(
            "  여론조사 분석: %d회 조사, %d명 추이",
            verdict.poll_analysis.total_surveys,
            len(verdict.poll_analysis.candidate_trends),
        )
    logger.info("=" * 60)

    gl.log_graph_end(
        district_name=district["name"],
        retry_count=retry_count,
        unresolved_errors=errors,
        candidates=_candidates_summary(verdict),
        analysis_mode=mode,
    )

    return verdict
