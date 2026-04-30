from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

from ingestion.base_registry import ComponentRegistry
from models.score import CandidateScore, DailyVerdict, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 한국 선거 판세 분석 전문가입니다.
제공된 뉴스 기사 청크를 분석하여 각 후보의 판세를 판정하고, 후보별로 판세를 끌어올리기 위한 전략까지 도출합니다.

판정 기준:
- "우세": 여론조사 선두, 긍정적 언론 보도 다수, 지지 기반 강화 징후
- "균형": 오차범위 내 접전, 혼재된 신호
- "열세": 여론조사 하위, 부정적 보도, 지지 기반 약화 징후

추가 분석:
- --district pyeongtaek_b 일 경우, 김용남: 현재 판세를 기반으로 지지율을 끌어올리기 위한 구체적 전략 제시
- --district busan_bukgu_gap 일 경우, 하정우: 현재 판세를 기반으로 지지율을 끌어올리기 위한 구체적 전략 제시
- 전략은 기사 내용(지지층, 이슈, 지역 기반, 경쟁 후보 상황 등)에 근거하여 현실적으로 작성

규칙:
- 모든 후보의 win_probability 합계는 반드시 1.0
- 여론조사 수치가 있으면 이를 기반으로, 없으면 기사 논조를 종합
- reasoning은 한국어로 2~3문장, 근거를 구체적으로 제시
- strategy는 한국어로 2~3문장, 실행 가능한 수준으로 작성


반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요:
{
  "candidates": [
    {
      "candidate": "후보명",
      "verdict": "우세|균형|열세",
      "win_probability": 0.0,
      "reasoning": "판정 근거"
    }
  ],
  "summary": "선거구 전체 판세 요약 (1~2문장)"
  "strategy": "전략"
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
        scores.append(CandidateScore(
            candidate=name,
            party=candidate_parties.get(name, ""),
            district_id=district["id"],
            verdict=item["verdict"],
            win_probability=item["win_probability"],
            reasoning=item["reasoning"],
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
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("[%s] LLM 응답 파싱 실패: %s — 균등 확률 배분", self.name, e)
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
