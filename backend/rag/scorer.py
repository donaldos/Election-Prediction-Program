from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

from ingestion.base_registry import ComponentRegistry
from models.score import CandidateReasoning, CandidateScore, DailyVerdict, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 한국 선거 판세 분석 전문가입니다.
제공된 뉴스 기사 청크와 여론조사 데이터를 분석하여 각 후보의 판세를 판정하고, 후보별로 9가지 분석 항목을 도출합니다.

판정 기준:
- "우세": 여론조사 선두, 긍정적 언론 보도 다수, 지지 기반 강화 징후
- "균형": 오차범위 내 접전, 혼재된 신호
- "열세": 여론조사 하위, 부정적 보도, 지지 기반 약화 징후

추가 분석:
- --district pyeongtaek_b 일 경우, 김용남: 현재 판세를 기반으로 지지율을 끌어올리기 위한 구체적 전략 제시
- --district busan_bukgu_gap 일 경우, 하정우: 현재 판세를 기반으로 지지율을 끌어올리기 위한 구체적 전략 제시

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
        "strategy": "선거 전략 — 지지율을 끌어올리기 위한 구체적 전략 (지지층 결집, 이슈 선점, 지역 공략 등). 기사 내용에 근거하여 실행 가능한 수준으로 5~7문장 작성",
        "forecast": "예측 — 향후 판세 전망 (추세 변화, 변수, 당선 가능성 근거)"
      }
    }
  ],
  "summary": "선거구 전체 판세 요약 (1~2문장)"
}"""


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
        lines.append("## 최신 여론조사")
        lines.append(f"조사기관: {poll_summary.pollster}, 조사일: {poll_summary.survey_date}")
        for c in poll_summary.candidates:
            lines.append(f"- {c.candidate} ({c.party}): {c.support}%")
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
    ) -> DailyVerdict:
        if not chunks:
            logger.warning("검색 결과 0건 — 균등 확률 배분")
            return self._empty_verdict(district)

        user_prompt = _build_user_prompt(chunks, district)

        logger.info(
            "[%s] LLM 판정 요청 — %s, 청크 %d건",
            self.name, district["name"], len(chunks),
        )
        logger.debug("[%s] 프롬프트:\n%s", self.name, user_prompt)

        t0 = time.monotonic()
        raw_response = self._call_llm(SYSTEM_PROMPT, user_prompt)
        elapsed = time.monotonic() - t0
        logger.info("[%s] LLM 응답 수신 — %.1f초", self.name, elapsed)
        logger.debug("[%s] LLM 응답:\n%s", self.name, raw_response)

        try:
            verdict = _parse_llm_response(raw_response, district, chunks)
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
