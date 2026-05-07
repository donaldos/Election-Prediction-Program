"""여론조사 데이터 영속 저장소 — JSONL 파일 기반."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

from models.poll import PollCandidateSupport, PollEntry, PollSummary

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POLLS_PATH = DATA_DIR / "polls.jsonl"


class PollStore:

    def __init__(self, path: Path = POLLS_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_id(entry: PollEntry) -> str:
        key = f"{entry.district_id}:{entry.candidate}:{entry.pollster}:{entry.survey_date}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def save(self, entries: list[PollEntry]) -> list[PollEntry]:
        existing = self.load_all()
        existing_ids = {e.id for e in existing}

        new_entries: list[PollEntry] = []
        for entry in entries:
            entry.id = self._make_id(entry)
            if entry.id in existing_ids:
                existing = [e for e in existing if e.id != entry.id]
            existing.append(entry)
            new_entries.append(entry)

        self._write_all(existing)
        logger.info("여론조사 저장 — %d건 (신규/갱신 %d건)", len(existing), len(new_entries))
        return new_entries

    def load_all(self) -> list[PollEntry]:
        if not self._path.exists():
            return []

        entries: list[PollEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(PollEntry.model_validate_json(line))
                except Exception as e:
                    logger.warning("여론조사 파싱 실패 — %s", e)
        return entries

    def load_by_district(self, district_id: str) -> list[PollEntry]:
        return [e for e in self.load_all() if e.district_id == district_id]

    def delete(self, entry_id: str) -> bool:
        entries = self.load_all()
        filtered = [e for e in entries if e.id != entry_id]
        if len(filtered) == len(entries):
            return False
        self._write_all(filtered)
        logger.info("여론조사 삭제 — id=%s", entry_id)
        return True

    def delete_all(self, district_id: str | None = None) -> int:
        entries = self.load_all()
        if district_id:
            filtered = [e for e in entries if e.district_id != district_id]
        else:
            filtered = []
        deleted = len(entries) - len(filtered)
        self._write_all(filtered)
        return deleted

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

    def _write_all(self, entries: list[PollEntry]) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")
