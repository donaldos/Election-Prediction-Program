"""여론조사 저장소 — Strategy + Registry 패턴.

config.yaml의 polls.type 값으로 구현체를 선택한다.
  - jsonl: JSONL 파일 기반 (기본값)
  - google_sheets: Google Sheets 읽기 전용
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

from ingestion.base_registry import ComponentRegistry
from models.poll import PollCandidateSupport, PollEntry, PollSummary

logger = logging.getLogger(__name__)


class AbstractPollStore(ABC):

    @staticmethod
    def _make_id(entry: PollEntry) -> str:
        key = f"{entry.district_id}:{entry.candidate}:{entry.pollster}:{entry.survey_date}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @abstractmethod
    def save(self, entries: list[PollEntry]) -> list[PollEntry]: ...

    @abstractmethod
    def load_all(self) -> list[PollEntry]: ...

    def load_by_district(self, district_id: str) -> list[PollEntry]:
        return [e for e in self.load_all() if e.district_id == district_id]

    @abstractmethod
    def delete(self, entry_id: str) -> bool: ...

    @abstractmethod
    def delete_all(self, district_id: str | None = None) -> int: ...

    def load_meta(self) -> list:
        """조사 메타데이터 로드 (Google Sheets 전용, 기본: 빈 리스트)."""
        return []

    def get_latest_summary(self, district_id: str) -> PollSummary | None:
        entries = self.load_by_district(district_id)
        if not entries:
            return None

        latest_date = max(e.survey_date for e in entries)
        latest = [e for e in entries if e.survey_date == latest_date]
        pollsters = {e.pollster for e in latest}
        pollster = ", ".join(sorted(pollsters))

        return PollSummary(
            district_id=district_id,
            pollster=pollster,
            survey_date=latest_date,
            candidates=[
                PollCandidateSupport(
                    candidate=e.candidate,
                    party=e.party,
                    support=e.support,
                )
                for e in sorted(latest, key=lambda x: x.support, reverse=True)
            ],
        )


PollStoreRegistry = ComponentRegistry(AbstractPollStore, "PollStore")


def create_poll_store(config: dict | None = None) -> AbstractPollStore:
    """config.yaml 기반 PollStore 인스턴스 생성."""
    import rag.jsonl_poll_store  # noqa: F401
    import rag.gsheets_poll_store  # noqa: F401

    if config is None:
        return PollStoreRegistry.create("jsonl")

    polls_cfg = config.get("polls", {})
    poll_type = polls_cfg.get("type", "jsonl")
    params = polls_cfg.get("params", {})
    return PollStoreRegistry.create(poll_type, **params)
