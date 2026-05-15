from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

from ingestion.base_registry import ComponentRegistry
from models.score import (
    CandidateComparison,
    CandidateDiagnosis,
    CandidateReasoning,
    CandidateScore,
    CandidateStrategy,
    ComparisonDimension,
    DailyVerdict,
    SearchResult,
)

logger = logging.getLogger(__name__)

VERDICT_PROMPT = """\
당신은 한국 선거 판세 분석 전문가입니다.
제공된 뉴스 기사 청크와 여론조사 데이터를 분석하여 각 후보의 판세를 판정하고, 후보별로 9가지 분석 항목을 도출합니다.

판정 기준 (여론조사 오차범위 ±3%p 적용):
- 여론조사 지지율은 ±3%p의 오차범위를 가짐. 두 후보 간 격차가 6%p 이내이면 통계적으로 동률(오차범위 중첩)로 간주
- "우세": 1위 후보와의 격차가 6%p 초과로 선두이며, 긍정적 언론 보도 다수, 지지 기반 강화 징후
- "균형": 1위 후보와의 격차가 6%p 이내 (오차범위 중첩 구간), 혼재된 신호
- "열세": 1위 후보와의 격차가 6%p 초과로 뒤처지며, 부정적 보도, 지지 기반 약화 징후
- 여론조사 수치만으로 판정하지 말고, 오차범위 내에서는 기사 논조·추세·이슈 등을 종합하여 최종 판정

규칙:
- 모든 후보의 win_probability 합계는 반드시 1.0
- 여론조사 수치가 있으면 이를 기반으로, 없으면 기사 논조를 종합
- reasoning 내부의 각 항목은 한국어로 3~5문장씩, 기사 내용에 근거하여 구체적으로 작성

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요:
{
  "candidates": [
    {
      "candidate": "후보명",
      "verdict": "우세|균형|열세",
      "win_probability": 0.0,
      "reasoning": {
        "support_rate": "지지율 — 최신 여론조사 수치, 정당 지지율 대비 개인 지지율, 타 후보와의 격차",
        "pledge_reaction": "공약 반응 — 핵심 공약에 대한 유권자·언론 반응, 실현 가능성 평가, 공약 차별성",
        "strengths": "강점 — 해당 후보에게 유리한 요인 (조직력, 인지도, 지지 기반, 긍정 보도 등)",
        "weaknesses": "약점 — 해당 후보에게 불리한 요인 (부정 보도, 내부 갈등, 약한 인지도, 리스크 등)",
        "issues": "이슈 — 후보 관련 주요 이슈·논란·쟁점 (스캔들, 정책 논쟁, 당내 갈등 등)",
        "support_trend": "지지율 추이 — 시간 경과에 따른 지지율 변화 방향 (상승세/하락세/정체), 변곡점과 원인",
        "public_opinion": "출마 여론 — 출마에 대한 여론 반응, 지역구 민심, 당내·당외 지지 수준",
        "strategy": "선거 전략 — 지지율을 끌어올리기 위한 구체적 전략 5~7문장",
        "forecast": "예측 — 향후 판세 전망 (추세 변화, 변수, 당선 가능성 근거)"
      }
    }
  ],
  "summary": "선거구 전체 판세 요약 (1~2문장)"
}"""

DIAGNOSIS_PROMPT = """\
당신은 한국 선거 전략 컨설턴트입니다.
아래에 제공되는 판세 판정 결과와 뉴스 기사 청크를 기반으로, 각 후보의 **문제점을 진단**합니다.

분석 원칙:
- 판정 결과(verdict, win_probability)를 사실로 수용하고, 그 원인을 분석
- 기사 청크에 근거하여 구체적인 문제점과 근본 원인을 도출
- severity는 "심각", "주의", "경미" 중 선택

반드시 아래 JSON 형식으로만 응답하세요:
{
  "diagnosis": [
    {
      "candidate": "후보명",
      "party": "정당명",
      "problems": ["문제점 1 (3~5문장)", "문제점 2 (3~5문장)"],
      "root_causes": ["근본 원인 1", "근본 원인 2"],
      "severity": "심각|주의|경미",
      "summary": "종합 진단 (2~3문장)"
    }
  ],
  "summary": "선거구 전체 문제점 진단 요약 (1~2문장)"
}"""

STRATEGY_PROMPT = """\
당신은 한국 선거 전략 컨설턴트입니다.
아래에 제공되는 판세 판정 결과, 문제점 진단, 뉴스 기사 청크를 기반으로 각 후보의 **대응방안과 전략**을 도출합니다.

분석 원칙:
- 진단된 문제점에 대한 구체적이고 실행 가능한 해결책 제시
- 기사에 언급된 실제 이슈와 지역 특성을 반영
- priority는 "긴급", "중요", "보통" 중 선택
- 각 action_plan 항목은 시간순으로 정렬하여 단기(1주)→중기(2~3주)→장기(선거일까지)로 구분

반드시 아래 JSON 형식으로만 응답하세요:
{
  "strategies": [
    {
      "candidate": "후보명",
      "party": "정당명",
      "solutions": ["해결방안 1 (3~5문장)", "해결방안 2 (3~5문장)"],
      "action_plan": ["단기: 구체적 실행 계획", "중기: 구체적 실행 계획", "장기: 구체적 실행 계획"],
      "priority": "긴급|중요|보통",
      "expected_impact": "예상 효과 (2~3문장)",
      "summary": "전략 종합 (2~3문장)"
    }
  ],
  "summary": "선거구 전체 전략 요약 (1~2문장)"
}"""

COMPARISON_PROMPT = """\
당신은 한국 선거 비교 분석 전문가입니다.
아래에 제공되는 판세 판정 결과와 뉴스 기사 청크를 기반으로, 지정된 두 후보를 **다차원 비교 분석**합니다.

비교 항목:
1. 지지율 및 여론조사 — 수치 비교, 추세, 오차범위 고려
2. 공약 및 정책 — 핵심 공약 비교, 유권자 반응 차이
3. 조직력 및 캠프 — 선거 조직, 당 지원, 자금력
4. 언론 및 여론 — 보도 논조, SNS 반응, 이미지
5. 강점 vs 약점 — 각 후보의 상대적 유불리
6. 지역 기반 — 지역구 내 지지 기반 및 텃밭 분석

분석 원칙:
- advantage는 "A 우위", "B 우위", "대등" 중 선택 (A, B는 실제 후보명으로 대체)
- overall_edge는 종합적으로 어느 후보가 우위인지 명시

반드시 아래 JSON 형식으로만 응답하세요:
{
  "comparisons": [
    {
      "candidate_a": "후보명A",
      "candidate_b": "후보명B",
      "dimensions": [
        {
          "dimension": "비교 항목명",
          "candidate_a_assessment": "후보A 평가 (3~5문장)",
          "candidate_b_assessment": "후보B 평가 (3~5문장)",
          "advantage": "A 우위|B 우위|대등"
        }
      ],
      "overall_edge": "종합 우위 판단 (2~3문장)",
      "summary": "비교 분석 요약 (2~3문장)"
    }
  ],
  "summary": "전체 비교 분석 요약 (1~2문장)"
}"""

SYSTEM_PROMPT = VERDICT_PROMPT


_CATEGORY_ORDER = [
    "지지율", "공약 반응", "강점", "약점",
    "이슈", "지지율 추이", "출마 여론", "선거 전략",
]


def flatten_grouped_chunks(
    grouped: dict[str, dict[str, list[SearchResult]]],
) -> list[SearchResult]:
    """그룹별 청크를 고유한 평탄 리스트로 변환."""
    seen: set[str] = set()
    result: list[SearchResult] = []
    for categories in grouped.values():
        for chunks in categories.values():
            for chunk in chunks:
                if chunk.id not in seen:
                    result.append(chunk)
                    seen.add(chunk.id)
    return result


MAX_CHUNKS_PER_CATEGORY = 3
TEXT_PREVIEW_LEN = 200


def _format_chunks_section(chunks: list[SearchResult], limit: int = MAX_CHUNKS_PER_CATEGORY) -> list[str]:
    lines: list[str] = []
    for i, chunk in enumerate(chunks[:limit], 1):
        lines.append(
            f"[{i}] {chunk.title} ({chunk.source}, {chunk.published_at:%Y-%m-%d}) "
            f"— score: {chunk.score:.3f}"
        )
        text_preview = chunk.text[:TEXT_PREVIEW_LEN].replace("\n", " ")
        lines.append(f"    {text_preview}")
        lines.append("")
    return lines


def _build_user_prompt_grouped(
    grouped: dict[str, dict[str, list[SearchResult]]],
    district: dict,
) -> str:
    lines = [f"## 선거구: {district['name']}\n"]

    lines.append("## 후보 목록")
    for cand in district.get("candidates", []):
        lines.append(f"- {cand['name']} ({cand['party']})")
    lines.append("")

    from rag.poll_store import PollStore
    poll_summary = PollStore().get_latest_summary(district["id"])
    if poll_summary:
        lines.append("## 최신 여론조사 (오차범위 ±3%p)")
        lines.append(f"조사기관: {poll_summary.pollster}, 조사일: {poll_summary.survey_date}")
        for c in poll_summary.candidates:
            low = round(c.support - 3.0, 1)
            high = round(c.support + 3.0, 1)
            lines.append(f"- {c.candidate} ({c.party}): {c.support}% (오차범위: {low}%~{high}%)")
        lines.append("※ 두 후보 간 격차 6%p 이내는 통계적 동률 (오차범위 중첩)")
        lines.append("")

    common = grouped.get("_common", {})
    if common:
        lines.append("## 공통 판세 자료\n")
        for category in sorted(common.keys()):
            chunks = common[category]
            lines.append(f"### {category} (상위 {min(len(chunks), MAX_CHUNKS_PER_CATEGORY)}/{len(chunks)}건)")
            lines.extend(_format_chunks_section(chunks))

    lines.append("## 후보별 분석 자료\n")
    for cand in district.get("candidates", []):
        name = cand["name"]
        party = cand["party"]
        candidate_data = grouped.get(name, {})
        if not candidate_data:
            continue

        lines.append(f"### {name} ({party})\n")

        for category in _CATEGORY_ORDER:
            chunks = candidate_data.get(category, [])
            if not chunks:
                continue
            lines.append(f"#### {category} (상위 {min(len(chunks), MAX_CHUNKS_PER_CATEGORY)}/{len(chunks)}건)")
            lines.extend(_format_chunks_section(chunks))

        for category, chunks in candidate_data.items():
            if category not in _CATEGORY_ORDER and chunks:
                lines.append(f"#### {category} (상위 {min(len(chunks), MAX_CHUNKS_PER_CATEGORY)}/{len(chunks)}건)")
                lines.extend(_format_chunks_section(chunks))

    lines.append(
        "## 요청\n"
        "위 분석 항목별로 분류된 근거를 참고하여 각 후보의 verdict, win_probability, reasoning을 "
        "JSON으로 출력하세요.\n"
        "각 reasoning 필드(support_rate, pledge_reaction, strengths 등)는 해당 분석 항목에 "
        "배치된 근거 기사를 우선적으로 참고하여 구체적으로 작성하세요."
    )
    return "\n".join(lines)


def _build_user_prompt(
    chunks: list[SearchResult],
    district: dict,
) -> str:
    lines = [f"## 선거구: {district['name']}\n"]

    lines.append("## 후보 목록")
    for cand in district.get("candidates", []):
        lines.append(f"- {cand['name']} ({cand['party']})")
    lines.append("")

    from rag.poll_store import PollStore
    poll_summary = PollStore().get_latest_summary(district["id"])
    if poll_summary:
        lines.append("## 최신 여론조사 (오차범위 ±3%p)")
        lines.append(f"조사기관: {poll_summary.pollster}, 조사일: {poll_summary.survey_date}")
        for c in poll_summary.candidates:
            low = round(c.support - 3.0, 1)
            high = round(c.support + 3.0, 1)
            lines.append(f"- {c.candidate} ({c.party}): {c.support}% (오차범위: {low}%~{high}%)")
        lines.append("※ 두 후보 간 격차 6%p 이내는 통계적 동률 (오차범위 중첩)")
        lines.append("")

    lines.append(f"## 수집된 뉴스 청크 ({len(chunks)}건)\n")
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {chunk.title} ({chunk.source}, {chunk.published_at:%Y-%m-%d}) "
            f"— score: {chunk.score:.3f}"
        )
        text_preview = chunk.text[:300].replace("\n", " ")
        lines.append(f"    {text_preview}")
        lines.append("")

    lines.append(
        "## 요청\n"
        "위 뉴스를 종합 분석하여 각 후보의 verdict, win_probability, reasoning을 "
        "JSON으로 출력하세요."
    )
    return "\n".join(lines)


def _parse_llm_response(
    raw: str,
    district: dict,
    chunks: list[SearchResult],
) -> DailyVerdict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)

    candidate_parties = {
        c["name"]: c["party"] for c in district.get("candidates", [])
    }
    chunk_ids = [c.id for c in chunks]

    scores: list[CandidateScore] = []
    for item in data.get("candidates", []):
        name = item["candidate"]
        raw_reasoning = item["reasoning"]
        if isinstance(raw_reasoning, dict):
            reasoning = CandidateReasoning(**raw_reasoning)
        else:
            reasoning = raw_reasoning
        scores.append(CandidateScore(
            candidate=name,
            party=candidate_parties.get(name, ""),
            district_id=district["id"],
            verdict=item["verdict"],
            win_probability=item["win_probability"],
            reasoning=reasoning,
            supporting_chunks=chunk_ids,
            chunk_count=len(chunks),
        ))

    scores = _normalize_probabilities(scores)

    return DailyVerdict(
        district_id=district["id"],
        district_name=district["name"],
        date=datetime.now(),
        candidates=scores,
        total_chunks_analyzed=len(chunks),
        summary=data.get("summary", ""),
    )


def _normalize_probabilities(scores: list[CandidateScore]) -> list[CandidateScore]:
    total = sum(s.win_probability for s in scores)
    if total == 0:
        equal = 1.0 / len(scores) if scores else 0
        for s in scores:
            s.win_probability = round(equal, 4)
    elif abs(total - 1.0) > 0.01:
        logger.warning("확률 합 %.4f ≠ 1.0 — 정규화 수행", total)
        for s in scores:
            s.win_probability = round(s.win_probability / total, 4)
    return scores


class AbstractScorer(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _call_llm(self, system: str, user: str, *, json_mode: bool = True) -> str:
        ...

    def score(
        self,
        chunks: list[SearchResult],
        district: dict,
        *,
        grouped_chunks: dict[str, dict[str, list[SearchResult]]] | None = None,
    ) -> DailyVerdict:
        if grouped_chunks is not None:
            effective_chunks = flatten_grouped_chunks(grouped_chunks)
            user_prompt = _build_user_prompt_grouped(grouped_chunks, district)
        else:
            effective_chunks = chunks
            user_prompt = _build_user_prompt(chunks, district)

        if not effective_chunks:
            logger.warning("검색 결과 0건 — 균등 확률 배분")
            return self._empty_verdict(district)

        logger.info(
            "[%s] LLM 판정 요청 — %s, 청크 %d건%s",
            self.name, district["name"], len(effective_chunks),
            " (그룹별 구조)" if grouped_chunks else "",
        )
        logger.debug("[%s] 프롬프트:\n%s", self.name, user_prompt)

        t0 = time.monotonic()
        raw_response = self._call_llm(SYSTEM_PROMPT, user_prompt)
        elapsed = time.monotonic() - t0
        logger.info("[%s] LLM 응답 수신 — %.1f초", self.name, elapsed)
        logger.debug("[%s] LLM 응답:\n%s", self.name, raw_response)

        try:
            verdict = _parse_llm_response(raw_response, district, effective_chunks)
        except (json.JSONDecodeError, KeyError, ValueError, Exception) as e:
            logger.warning("[%s] LLM 응답 파싱 실패: %s — 균등 확률 배분", self.name, e)
            logger.debug("[%s] 파싱 실패 원문:\n%s", self.name, raw_response)
            return self._empty_verdict(district)

        logger.info(
            "[%s] 판정 완료 — %s: %s",
            self.name,
            district["name"],
            ", ".join(f"{s.candidate}={s.verdict}({s.win_probability:.1%})" for s in verdict.candidates),
        )
        return verdict

    def diagnose(
        self,
        verdict: DailyVerdict,
        chunks: list[SearchResult],
        district: dict,
        *,
        target_candidates: list[str] | None = None,
    ) -> list[CandidateDiagnosis]:
        user_prompt = _build_diagnosis_user_prompt(verdict, chunks, district, target_candidates)

        logger.info("[%s] 문제점 진단 요청 — %s", self.name, district["name"])
        t0 = time.monotonic()
        raw = self._call_llm(DIAGNOSIS_PROMPT, user_prompt)
        elapsed = time.monotonic() - t0
        logger.info("[%s] 진단 응답 수신 — %.1f초", self.name, elapsed)

        try:
            return _parse_diagnosis_response(raw)
        except Exception as e:
            logger.warning("[%s] 진단 파싱 실패: %s", self.name, e)
            return []

    def strategize(
        self,
        verdict: DailyVerdict,
        diagnosis: list[CandidateDiagnosis],
        chunks: list[SearchResult],
        district: dict,
        *,
        target_candidates: list[str] | None = None,
    ) -> list[CandidateStrategy]:
        user_prompt = _build_strategy_user_prompt(
            verdict, diagnosis, chunks, district, target_candidates,
        )

        logger.info("[%s] 전략 도출 요청 — %s", self.name, district["name"])
        t0 = time.monotonic()
        raw = self._call_llm(STRATEGY_PROMPT, user_prompt)
        elapsed = time.monotonic() - t0
        logger.info("[%s] 전략 응답 수신 — %.1f초", self.name, elapsed)

        try:
            return _parse_strategy_response(raw)
        except Exception as e:
            logger.warning("[%s] 전략 파싱 실패: %s", self.name, e)
            return []

    def compare(
        self,
        verdict: DailyVerdict,
        chunks: list[SearchResult],
        district: dict,
        candidate_a: str,
        candidate_b: str,
    ) -> list[CandidateComparison]:
        user_prompt = _build_comparison_user_prompt(
            verdict, chunks, district, candidate_a, candidate_b,
        )

        logger.info(
            "[%s] 비교 분석 요청 — %s vs %s", self.name, candidate_a, candidate_b,
        )
        t0 = time.monotonic()
        raw = self._call_llm(COMPARISON_PROMPT, user_prompt)
        elapsed = time.monotonic() - t0
        logger.info("[%s] 비교 응답 수신 — %.1f초", self.name, elapsed)

        try:
            return _parse_comparison_response(raw)
        except Exception as e:
            logger.warning("[%s] 비교 파싱 실패: %s", self.name, e)
            return []

    @staticmethod
    def _empty_verdict(district: dict) -> DailyVerdict:
        candidates = district.get("candidates", [])
        equal = round(1.0 / len(candidates), 4) if candidates else 0
        return DailyVerdict(
            district_id=district["id"],
            district_name=district["name"],
            date=datetime.now(),
            candidates=[
                CandidateScore(
                    candidate=c["name"],
                    party=c["party"],
                    district_id=district["id"],
                    verdict="균형",
                    win_probability=equal,
                    reasoning="분석 가능한 데이터가 부족합니다.",
                    supporting_chunks=[],
                    chunk_count=0,
                )
                for c in candidates
            ],
            total_chunks_analyzed=0,
            summary="데이터 부족으로 판정을 보류합니다.",
        )


ScorerRegistry = ComponentRegistry(AbstractScorer, "Scorer")


# ---------------------------------------------------------------------------
# 분석 모드별 유저 프롬프트 빌더 + 파서
# ---------------------------------------------------------------------------

def _verdict_context_block(verdict: DailyVerdict) -> str:
    lines = ["## 판정 결과 (score 노드 출력)"]
    for s in verdict.candidates:
        lines.append(f"- {s.candidate} ({s.party}): {s.verdict}, 승률 {s.win_probability:.1%}")
    lines.append(f"요약: {verdict.summary}")
    return "\n".join(lines)


def _chunks_context_block(chunks: list[SearchResult], limit: int = 10) -> str:
    lines = [f"## 참고 뉴스 청크 (상위 {min(len(chunks), limit)}/{len(chunks)}건)"]
    for i, c in enumerate(chunks[:limit], 1):
        text_preview = c.text[:200].replace("\n", " ")
        lines.append(f"[{i}] {c.title} ({c.source}, {c.published_at:%Y-%m-%d})")
        lines.append(f"    {text_preview}")
        lines.append("")
    return "\n".join(lines)


def _build_diagnosis_user_prompt(
    verdict: DailyVerdict,
    chunks: list[SearchResult],
    district: dict,
    target_candidates: list[str] | None,
) -> str:
    lines = [f"## 선거구: {district['name']}\n"]
    lines.append(_verdict_context_block(verdict))
    lines.append("")
    lines.append(_chunks_context_block(chunks))
    if target_candidates:
        lines.append(f"\n## 분석 대상 후보: {', '.join(target_candidates)}")
        lines.append("위 후보에 대해서만 문제점을 진단하세요.")
    else:
        lines.append("\n모든 후보에 대해 문제점을 진단하세요.")
    return "\n".join(lines)


def _build_strategy_user_prompt(
    verdict: DailyVerdict,
    diagnosis: list[CandidateDiagnosis],
    chunks: list[SearchResult],
    district: dict,
    target_candidates: list[str] | None,
) -> str:
    lines = [f"## 선거구: {district['name']}\n"]
    lines.append(_verdict_context_block(verdict))
    lines.append("")
    lines.append("## 문제점 진단 결과 (diagnose 노드 출력)")
    for d in diagnosis:
        lines.append(f"### {d.candidate} ({d.party}) — 심각도: {d.severity}")
        for i, p in enumerate(d.problems, 1):
            lines.append(f"  문제 {i}: {p}")
        for i, r in enumerate(d.root_causes, 1):
            lines.append(f"  원인 {i}: {r}")
        lines.append("")
    lines.append(_chunks_context_block(chunks))
    if target_candidates:
        lines.append(f"\n## 분석 대상 후보: {', '.join(target_candidates)}")
        lines.append("위 후보에 대해서만 대응방안과 전략을 도출하세요.")
    else:
        lines.append("\n모든 후보에 대해 대응방안과 전략을 도출하세요.")
    return "\n".join(lines)


def _build_comparison_user_prompt(
    verdict: DailyVerdict,
    chunks: list[SearchResult],
    district: dict,
    candidate_a: str,
    candidate_b: str,
) -> str:
    lines = [f"## 선거구: {district['name']}\n"]
    lines.append(_verdict_context_block(verdict))
    lines.append("")
    lines.append(_chunks_context_block(chunks))
    lines.append(f"\n## 비교 대상: {candidate_a} vs {candidate_b}")
    lines.append("위 두 후보를 6가지 차원에서 비교 분석하세요.")
    return "\n".join(lines)


def _strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_diagnosis_response(raw: str) -> list[CandidateDiagnosis]:
    data = json.loads(_strip_markdown_fence(raw))
    return [CandidateDiagnosis(**item) for item in data.get("diagnosis", [])]


def _parse_strategy_response(raw: str) -> list[CandidateStrategy]:
    data = json.loads(_strip_markdown_fence(raw))
    return [CandidateStrategy(**item) for item in data.get("strategies", [])]


def _parse_comparison_response(raw: str) -> list[CandidateComparison]:
    data = json.loads(_strip_markdown_fence(raw))
    results = []
    for item in data.get("comparisons", []):
        dims = [ComparisonDimension(**d) for d in item.get("dimensions", [])]
        results.append(CandidateComparison(
            candidate_a=item["candidate_a"],
            candidate_b=item["candidate_b"],
            dimensions=dims,
            overall_edge=item["overall_edge"],
            summary=item["summary"],
        ))
    return results
