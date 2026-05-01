"""판정 결과 영속 저장소 — JSONL 파일 기반."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from models.score import DailyVerdict

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VERDICTS_DIR = DATA_DIR / "verdicts"


class VerdictStore:

    def __init__(self, base_dir: Path = VERDICTS_DIR) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _district_path(self, district_id: str) -> Path:
        return self._base_dir / f"{district_id}.jsonl"

    def save(self, verdict: DailyVerdict) -> Path:
        path = self._district_path(verdict.district_id)
        line = json.dumps(verdict.model_dump(mode="json"), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        logger.info(
            "판정 저장 — %s (%s), 후보 %d명",
            verdict.district_name, verdict.date.strftime("%Y-%m-%d %H:%M"), len(verdict.candidates),
        )
        return path

    def load_all(self, district_id: str) -> list[DailyVerdict]:
        path = self._district_path(district_id)
        if not path.exists():
            return []

        verdicts: list[DailyVerdict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    verdicts.append(DailyVerdict.model_validate_json(line))
                except Exception as e:
                    logger.warning("판정 파싱 실패 — %s", e)
                    continue
        return verdicts

    def load_latest(self, district_id: str) -> DailyVerdict | None:
        verdicts = self.load_all(district_id)
        if not verdicts:
            return None
        return max(verdicts, key=lambda v: v.date)

    def load_range(
        self,
        district_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[DailyVerdict]:
        verdicts = self.load_all(district_id)
        if date_from:
            verdicts = [v for v in verdicts if v.date >= date_from]
        if date_to:
            verdicts = [v for v in verdicts if v.date <= date_to]
        return sorted(verdicts, key=lambda v: v.date)

    def list_districts(self) -> list[str]:
        return [
            p.stem for p in self._base_dir.glob("*.jsonl")
        ]

    def count(self, district_id: str) -> int:
        return len(self.load_all(district_id))
