from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from models.score import CandidateScore, DailyVerdict
from rag.verdict_store import VerdictStore


def _make_verdict(
    district_id: str = "pyeongtaek_b",
    district_name: str = "평택을",
    date: datetime | None = None,
    summary: str = "테스트 요약",
) -> DailyVerdict:
    return DailyVerdict(
        district_id=district_id,
        district_name=district_name,
        date=date or datetime(2026, 5, 1, 12, 0),
        candidates=[
            CandidateScore(
                candidate="후보A",
                party="A당",
                district_id=district_id,
                verdict="우세",
                win_probability=0.6,
                reasoning="여론조사 선두",
                supporting_chunks=["id1"],
                chunk_count=5,
            ),
            CandidateScore(
                candidate="후보B",
                party="B당",
                district_id=district_id,
                verdict="열세",
                win_probability=0.4,
                reasoning="여론조사 하위",
                supporting_chunks=["id2"],
                chunk_count=5,
            ),
        ],
        total_chunks_analyzed=10,
        summary=summary,
    )


class TestVerdictStore:

    @pytest.fixture
    def store(self, tmp_path):
        return VerdictStore(base_dir=tmp_path)

    def test_save_and_load(self, store):
        verdict = _make_verdict()
        store.save(verdict)
        loaded = store.load_all("pyeongtaek_b")
        assert len(loaded) == 1
        assert loaded[0].district_id == "pyeongtaek_b"
        assert loaded[0].candidates[0].candidate == "후보A"

    def test_save_multiple(self, store):
        store.save(_make_verdict(date=datetime(2026, 5, 1)))
        store.save(_make_verdict(date=datetime(2026, 5, 2)))
        store.save(_make_verdict(date=datetime(2026, 5, 3)))
        assert store.count("pyeongtaek_b") == 3

    def test_load_latest(self, store):
        store.save(_make_verdict(date=datetime(2026, 5, 1)))
        store.save(_make_verdict(date=datetime(2026, 5, 3)))
        store.save(_make_verdict(date=datetime(2026, 5, 2)))
        latest = store.load_latest("pyeongtaek_b")
        assert latest.date == datetime(2026, 5, 3)

    def test_load_latest_empty(self, store):
        assert store.load_latest("nonexistent") is None

    def test_load_all_empty(self, store):
        assert store.load_all("nonexistent") == []

    def test_load_range(self, store):
        store.save(_make_verdict(date=datetime(2026, 5, 1)))
        store.save(_make_verdict(date=datetime(2026, 5, 5)))
        store.save(_make_verdict(date=datetime(2026, 5, 10)))

        results = store.load_range(
            "pyeongtaek_b",
            date_from=datetime(2026, 5, 3),
            date_to=datetime(2026, 5, 7),
        )
        assert len(results) == 1
        assert results[0].date == datetime(2026, 5, 5)

    def test_load_range_no_filter(self, store):
        store.save(_make_verdict(date=datetime(2026, 5, 1)))
        store.save(_make_verdict(date=datetime(2026, 5, 2)))
        results = store.load_range("pyeongtaek_b")
        assert len(results) == 2

    def test_list_districts(self, store):
        store.save(_make_verdict(district_id="pyeongtaek_b"))
        store.save(_make_verdict(district_id="busan_bukgu_gap", district_name="부산북구갑"))
        districts = store.list_districts()
        assert set(districts) == {"pyeongtaek_b", "busan_bukgu_gap"}

    def test_list_districts_empty(self, store):
        assert store.list_districts() == []

    def test_count(self, store):
        assert store.count("pyeongtaek_b") == 0
        store.save(_make_verdict())
        assert store.count("pyeongtaek_b") == 1

    def test_different_districts_separate_files(self, store):
        store.save(_make_verdict(district_id="pyeongtaek_b"))
        store.save(_make_verdict(district_id="busan_bukgu_gap", district_name="부산북구갑"))
        assert store.count("pyeongtaek_b") == 1
        assert store.count("busan_bukgu_gap") == 1
