"""여론조사 저장소 테스트 — Strategy + Registry 패턴."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.poll import PollEntry
from rag.poll_store import AbstractPollStore, PollStoreRegistry, create_poll_store


# ── 테스트 데이터 ────────────────────────────────────

def _make_entry(
    district_id: str = "pyeongtaek_b",
    candidate: str = "김용남",
    party: str = "더불어민주당",
    support: float = 29.0,
    pollster: str = "뉴스1",
    survey_date: str = "2026-05-14",
) -> PollEntry:
    return PollEntry(
        district_id=district_id,
        candidate=candidate,
        party=party,
        support=support,
        pollster=pollster,
        survey_date=date.fromisoformat(survey_date),
    )


SAMPLE_ENTRIES = [
    _make_entry(candidate="김용남", support=29.0),
    _make_entry(candidate="조국", party="조국혁신당", support=24.0),
    _make_entry(candidate="유의동", party="국민의힘", support=20.0),
]

SAMPLE_BUSAN = [
    _make_entry(
        district_id="busan_bukgu_gap",
        candidate="하정우",
        support=38.0,
        pollster="SBS",
        survey_date="2026-05-03",
    ),
]


# ── Registry 테스트 ──────────────────────────────────

class TestPollStoreRegistry:

    def test_jsonl_registered(self):
        import rag.jsonl_poll_store  # noqa: F401
        assert "jsonl" in PollStoreRegistry.registered_names

    def test_google_sheets_registered(self):
        import rag.gsheets_poll_store  # noqa: F401
        assert "google_sheets" in PollStoreRegistry.registered_names

    def test_create_jsonl(self, tmp_path):
        import rag.jsonl_poll_store  # noqa: F401
        store = PollStoreRegistry.create("jsonl", path=str(tmp_path / "test.jsonl"))
        assert store is not None

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="미등록"):
            PollStoreRegistry.create("unknown_type")


# ── JsonlPollStore 테스트 ────────────────────────────

class TestJsonlPollStore:

    @pytest.fixture()
    def store(self, tmp_path):
        import rag.jsonl_poll_store  # noqa: F401
        return PollStoreRegistry.create("jsonl", path=str(tmp_path / "polls.jsonl"))

    def test_save_and_load(self, store):
        store.save(SAMPLE_ENTRIES)
        loaded = store.load_all()
        assert len(loaded) == 3
        assert all(e.id for e in loaded)

    def test_save_deduplicates(self, store):
        store.save(SAMPLE_ENTRIES)
        store.save(SAMPLE_ENTRIES)
        loaded = store.load_all()
        assert len(loaded) == 3

    def test_load_by_district(self, store):
        store.save(SAMPLE_ENTRIES + SAMPLE_BUSAN)
        pt = store.load_by_district("pyeongtaek_b")
        bs = store.load_by_district("busan_bukgu_gap")
        assert len(pt) == 3
        assert len(bs) == 1

    def test_delete(self, store):
        saved = store.save(SAMPLE_ENTRIES)
        target_id = saved[0].id
        assert store.delete(target_id) is True
        assert len(store.load_all()) == 2

    def test_delete_not_found(self, store):
        store.save(SAMPLE_ENTRIES)
        assert store.delete("nonexistent") is False

    def test_delete_all(self, store):
        store.save(SAMPLE_ENTRIES + SAMPLE_BUSAN)
        deleted = store.delete_all()
        assert deleted == 4
        assert len(store.load_all()) == 0

    def test_delete_all_by_district(self, store):
        store.save(SAMPLE_ENTRIES + SAMPLE_BUSAN)
        deleted = store.delete_all("pyeongtaek_b")
        assert deleted == 3
        assert len(store.load_all()) == 1

    def test_get_latest_summary(self, store):
        older = _make_entry(candidate="김용남", support=25.0, survey_date="2026-05-07")
        store.save(SAMPLE_ENTRIES + [older])
        summary = store.get_latest_summary("pyeongtaek_b")
        assert summary is not None
        assert summary.survey_date == date(2026, 5, 14)
        assert len(summary.candidates) == 3
        assert summary.candidates[0].support == 29.0

    def test_get_latest_summary_empty(self, store):
        assert store.get_latest_summary("nonexistent") is None

    def test_empty_file(self, store):
        assert store.load_all() == []

    def test_make_id_deterministic(self):
        e1 = _make_entry()
        e2 = _make_entry()
        assert AbstractPollStore._make_id(e1) == AbstractPollStore._make_id(e2)

    def test_make_id_different_for_different_entries(self):
        e1 = _make_entry(candidate="김용남")
        e2 = _make_entry(candidate="조국")
        assert AbstractPollStore._make_id(e1) != AbstractPollStore._make_id(e2)


# ── GoogleSheetsPollStore 테스트 ─────────────────────

class TestGoogleSheetsPollStore:

    @pytest.fixture()
    def mock_sheets_data(self):
        meta_rows = [
            {
                "survey_id": "p0514_news1",
                "date": "2026-05-14",
                "district_id": "pyeongtaek_b",
                "disctrict_name": "평택을",
                "pollster": "뉴스1",
                "sample_size": 804,
                "margin_of_error": 3.5,
                "method": "전화면접",
                "pubulisher": "리서치앤리서치",
                "url": "",
            },
        ]
        candidate_rows = [
            {
                "survey_id": "p0514_news1",
                "candidate": "김용남",
                "party": "더불어민주당",
                "support": 29,
            },
            {
                "survey_id": "p0514_news1",
                "candidate": "조국",
                "party": "조국혁신당",
                "support": 24,
            },
        ]
        return meta_rows, candidate_rows

    @pytest.fixture()
    def store(self, mock_sheets_data):
        import rag.gsheets_poll_store  # noqa: F401
        s = PollStoreRegistry.create(
            "google_sheets",
            spreadsheet_id="fake_id",
            credentials_path="fake.json",
        )

        meta_rows, candidate_rows = mock_sheets_data
        mock_sh = MagicMock()
        mock_meta_ws = MagicMock()
        mock_meta_ws.get_all_records.return_value = meta_rows
        mock_cand_ws = MagicMock()
        mock_cand_ws.get_all_records.return_value = candidate_rows
        mock_sh.worksheet.side_effect = lambda name: (
            mock_meta_ws if name == "polls_meta" else mock_cand_ws
        )

        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_sh
        s._client = mock_client
        return s

    def test_load_all(self, store):
        entries = store.load_all()
        assert len(entries) == 2
        assert entries[0].candidate == "김용남"
        assert entries[0].support == 29.0
        assert entries[0].district_id == "pyeongtaek_b"

    def test_load_by_district(self, store):
        entries = store.load_by_district("pyeongtaek_b")
        assert len(entries) == 2

    def test_load_by_district_empty(self, store):
        entries = store.load_by_district("nonexistent")
        assert entries == []

    def test_get_latest_summary(self, store):
        summary = store.get_latest_summary("pyeongtaek_b")
        assert summary is not None
        assert summary.pollster == "뉴스1"
        assert len(summary.candidates) == 2

    def test_save_raises(self, store):
        with pytest.raises(NotImplementedError, match="스프레드시트"):
            store.save([])

    def test_delete_raises(self, store):
        with pytest.raises(NotImplementedError, match="스프레드시트"):
            store.delete("some_id")

    def test_delete_all_raises(self, store):
        with pytest.raises(NotImplementedError, match="스프레드시트"):
            store.delete_all()

    def test_ids_are_generated(self, store):
        entries = store.load_all()
        assert all(e.id for e in entries)
        assert entries[0].id != entries[1].id

    def test_skips_invalid_date(self):
        import rag.gsheets_poll_store  # noqa: F401
        s = PollStoreRegistry.create(
            "google_sheets",
            spreadsheet_id="fake_id",
            credentials_path="fake.json",
        )

        meta_rows = [
            {
                "survey_id": "p_good",
                "date": "2026-05-14",
                "district_id": "pyeongtaek_b",
                "disctrict_name": "평택을",
                "pollster": "뉴스1",
                "sample_size": 804,
                "margin_of_error": 3.5,
                "method": "전화면접",
                "pubulisher": "",
                "url": "",
            },
            {
                "survey_id": "p_bad",
                "date": "invalid-date",
                "district_id": "pyeongtaek_b",
                "disctrict_name": "평택을",
                "pollster": "뉴스1",
                "sample_size": 500,
                "margin_of_error": 4.4,
                "method": "ARS",
                "pubulisher": "",
                "url": "",
            },
        ]
        candidate_rows = [
            {"survey_id": "p_good", "candidate": "김용남", "party": "더불어민주당", "support": 29},
            {"survey_id": "p_bad", "candidate": "테스트", "party": "무소속", "support": 10},
        ]

        mock_sh = MagicMock()
        mock_meta_ws = MagicMock()
        mock_meta_ws.get_all_records.return_value = meta_rows
        mock_cand_ws = MagicMock()
        mock_cand_ws.get_all_records.return_value = candidate_rows
        mock_sh.worksheet.side_effect = lambda name: (
            mock_meta_ws if name == "polls_meta" else mock_cand_ws
        )
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_sh
        s._client = mock_client

        entries = s.load_all()
        assert len(entries) == 1

    def test_skips_missing_fields(self):
        import rag.gsheets_poll_store  # noqa: F401
        s = PollStoreRegistry.create(
            "google_sheets",
            spreadsheet_id="fake_id",
            credentials_path="fake.json",
        )

        meta_rows = [
            {
                "survey_id": "p_good",
                "date": "2026-05-14",
                "district_id": "pyeongtaek_b",
                "disctrict_name": "평택을",
                "pollster": "뉴스1",
                "sample_size": 804,
                "margin_of_error": 3.5,
                "method": "전화면접",
                "pubulisher": "",
                "url": "",
            },
        ]
        candidate_rows = [
            {"survey_id": "p_good", "candidate": "김용남", "party": "더불어민주당", "support": 29},
            {"survey_id": "p_good", "candidate": "", "party": "", "support": 0},
            {"survey_id": "unknown_id", "candidate": "테스트", "party": "무소속", "support": 10},
        ]

        mock_sh = MagicMock()
        mock_meta_ws = MagicMock()
        mock_meta_ws.get_all_records.return_value = meta_rows
        mock_cand_ws = MagicMock()
        mock_cand_ws.get_all_records.return_value = candidate_rows
        mock_sh.worksheet.side_effect = lambda name: (
            mock_meta_ws if name == "polls_meta" else mock_cand_ws
        )
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_sh
        s._client = mock_client

        entries = s.load_all()
        assert len(entries) == 1


# ── create_poll_store 팩토리 테스트 ──────────────────

class TestCreatePollStore:

    def test_default_creates_jsonl(self):
        store = create_poll_store()
        from rag.jsonl_poll_store import JsonlPollStore
        assert isinstance(store, JsonlPollStore)

    def test_config_jsonl(self, tmp_path):
        config = {"polls": {"type": "jsonl", "params": {"path": str(tmp_path / "p.jsonl")}}}
        store = create_poll_store(config)
        from rag.jsonl_poll_store import JsonlPollStore
        assert isinstance(store, JsonlPollStore)

    def test_config_google_sheets(self):
        config = {
            "polls": {
                "type": "google_sheets",
                "params": {
                    "spreadsheet_id": "fake",
                    "credentials_path": "fake.json",
                },
            }
        }
        store = create_poll_store(config)
        from rag.gsheets_poll_store import GoogleSheetsPollStore
        assert isinstance(store, GoogleSheetsPollStore)

    def test_no_polls_section_defaults_to_jsonl(self):
        config = {"rag": {}}
        store = create_poll_store(config)
        from rag.jsonl_poll_store import JsonlPollStore
        assert isinstance(store, JsonlPollStore)
