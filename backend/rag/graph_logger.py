"""LangGraph 실행 구조화 로그 — JSONL 파일 기반.

실행 단위로 data/graph_logs/{district_id}_{timestamp}.jsonl 파일을 생성하고,
노드 전환마다 JSON 라인을 기록한다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH_LOGS_DIR = DATA_DIR / "graph_logs"


class GraphLogger:

    def __init__(self, district_id: str) -> None:
        GRAPH_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._path = GRAPH_LOGS_DIR / f"{district_id}_{ts}.jsonl"
        self._district_id = district_id
        self._start_time = datetime.now()
        self._step = 0

    @property
    def path(self) -> Path:
        return self._path

    def log(self, node: str, event: str, **data: Any) -> None:
        self._step += 1
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": self._step,
            "node": node,
            "event": event,
            "district_id": self._district_id,
        }
        record.update(data)

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_graph_start(
        self,
        district_name: str,
        chunk_count: int,
        max_retries: int,
        threshold: float,
        has_previous: bool,
        analysis_mode: str = "verdict",
        target_candidates: list[str] | None = None,
    ) -> None:
        self.log(
            node="graph",
            event="start",
            district_name=district_name,
            chunk_count=chunk_count,
            max_retries=max_retries,
            consistency_threshold=threshold,
            has_previous_verdict=has_previous,
            analysis_mode=analysis_mode,
            target_candidates=target_candidates or [],
        )

    def log_graph_end(
        self,
        district_name: str,
        retry_count: int,
        unresolved_errors: list[str],
        candidates: list[dict],
        analysis_mode: str = "verdict",
    ) -> None:
        elapsed = (datetime.now() - self._start_time).total_seconds()
        self.log(
            node="graph",
            event="end",
            district_name=district_name,
            retry_count=retry_count,
            unresolved_errors=unresolved_errors,
            candidates=candidates,
            elapsed_seconds=round(elapsed, 1),
            analysis_mode=analysis_mode,
        )
        logger.info("그래프 로그 저장 — %s (%d 스텝)", self._path.name, self._step)

    def log_score_enter(self, district_name: str, chunk_count: int, retry: int) -> None:
        self.log(
            node="score",
            event="enter",
            district_name=district_name,
            chunk_count=chunk_count,
            retry=retry,
        )

    def log_score_exit(self, candidates: list[dict]) -> None:
        self.log(node="score", event="exit", candidates=candidates)

    def log_validate_enter(self, has_previous: bool) -> None:
        self.log(node="validate", event="enter", has_previous_verdict=has_previous)

    def log_validate_exit(
        self,
        grounding: str,
        consistency: str,
        probability: str,
        grounding_errors: list[str],
        consistency_errors: list[str],
        probability_errors: list[str],
    ) -> None:
        self.log(
            node="validate",
            event="exit",
            grounding=grounding,
            consistency=consistency,
            probability=probability,
            grounding_errors=grounding_errors,
            consistency_errors=consistency_errors,
            probability_errors=probability_errors,
            total_errors=len(grounding_errors) + len(consistency_errors) + len(probability_errors),
        )

    def log_correct_enter(self, errors: list[str], retry_before: int, retry_after: int) -> None:
        self.log(
            node="correct",
            event="enter",
            errors=errors,
            retry_before=retry_before,
            retry_after=retry_after,
        )

    def log_route(self, decision: str, reason: str) -> None:
        self.log(node="route", event="decision", decision=decision, reason=reason)

    def log_analysis_enter(self, node_name: str, target: str) -> None:
        self.log(node=node_name, event="enter", target=target)

    def log_analysis_exit(self, node_name: str, result_count: int) -> None:
        self.log(node=node_name, event="exit", result_count=result_count)
