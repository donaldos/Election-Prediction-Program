from __future__ import annotations

import copy
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from models.score import CandidateScore, DailyVerdict


SAMPLE_CONFIG = {
    "districts": [
        {
            "id": "pyeongtaek_b",
            "name": "평택을",
            "candidates": [
                {"name": "후보A", "party": "A당", "keywords": ["후보A"]},
            ],
        },
    ],
    "scrapers": {
        "naver": {"type": "naver", "params": {"lookback_days": 2}},
        "political": {"type": "political", "params": {"urls": [], "lookback_days": 2}},
    },
    "chunker": {"type": "korean_paragraph", "params": {"chunk_size": 400}},
    "embedder": {"type": "openai", "params": {"model": "text-embedding-3-small", "dimensions": 1536}},
    "vectordb": {"type": "chroma", "collection": "test_chunks", "params": {"persist_dir": "/tmp/test_chroma"}},
    "rag": {
        "retriever": {"top_k": 20, "lookback_days": 14},
        "reranker": {"min_score": 0.3, "deduplicate": True},
        "scorer": {"provider": "openai", "model": "gpt-4o", "temperature": 0.1, "max_tokens": 2000},
        "purge_days": 60,
    },
}


SAMPLE_VERDICT = DailyVerdict(
    district_id="pyeongtaek_b",
    district_name="평택을",
    date=datetime(2026, 5, 1, 12, 0),
    candidates=[
        CandidateScore(
            candidate="후보A",
            party="A당",
            district_id="pyeongtaek_b",
            verdict="우세",
            win_probability=0.6,
            reasoning="선두",
            supporting_chunks=["id1"],
            chunk_count=5,
        ),
    ],
    total_chunks_analyzed=10,
    summary="테스트 요약",
)


@pytest.fixture
def client():
    with patch("app.core.dependencies.get_config", return_value=copy.deepcopy(SAMPLE_CONFIG)):
        with patch("app.core.dependencies.reload_config", return_value=copy.deepcopy(SAMPLE_CONFIG)):
            from app.main import app
            yield TestClient(app)


class TestGetLatestVerdict:
    def test_returns_latest(self, client):
        mock_store = MagicMock()
        mock_store.load_latest.return_value = SAMPLE_VERDICT
        with patch("rag.verdict_store.VerdictStore", return_value=mock_store):
            resp = client.get("/api/v1/scores/pyeongtaek_b/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["district_id"] == "pyeongtaek_b"
        assert data["candidates"][0]["candidate"] == "후보A"

    def test_not_found(self, client):
        mock_store = MagicMock()
        mock_store.load_latest.return_value = None
        with patch("rag.verdict_store.VerdictStore", return_value=mock_store):
            resp = client.get("/api/v1/scores/nonexistent/latest")
        assert resp.status_code == 404


class TestGetVerdictHistory:
    def test_returns_list(self, client):
        mock_store = MagicMock()
        mock_store.load_range.return_value = [SAMPLE_VERDICT, SAMPLE_VERDICT]
        with patch("rag.verdict_store.VerdictStore", return_value=mock_store):
            resp = client.get("/api/v1/scores/pyeongtaek_b/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["verdicts"]) == 2

    def test_empty_history(self, client):
        mock_store = MagicMock()
        mock_store.load_range.return_value = []
        with patch("rag.verdict_store.VerdictStore", return_value=mock_store):
            resp = client.get("/api/v1/scores/pyeongtaek_b/history")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestGetTimeseries:
    def test_returns_points(self, client):
        mock_store = MagicMock()
        mock_store.load_range.return_value = [SAMPLE_VERDICT]
        with patch("rag.verdict_store.VerdictStore", return_value=mock_store):
            resp = client.get("/api/v1/scores/pyeongtaek_b/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["district_name"] == "평택을"
        assert len(data["points"]) == 1
        assert "후보A" in data["points"][0]["candidates"]

    def test_unknown_district(self, client):
        resp = client.get("/api/v1/scores/unknown_district/timeseries")
        assert resp.status_code == 404
