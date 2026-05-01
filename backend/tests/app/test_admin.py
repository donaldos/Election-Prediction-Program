from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.pipeline_runner import PipelineRunner, TaskState, TaskStatus


SAMPLE_CONFIG = {
    "districts": [
        {
            "id": "pyeongtaek_b",
            "name": "평택을",
            "candidates": [
                {"name": "김용남", "party": "더불어민주당", "keywords": ["김용남"]},
            ],
        },
        {
            "id": "busan_bukgu_gap",
            "name": "부산북구갑",
            "candidates": [
                {"name": "한동훈", "party": "무소속", "keywords": ["한동훈"]},
            ],
        },
    ],
    "scrapers": {
        "naver": {"type": "naver", "params": {"max_articles_per_run": 10, "lookback_days": 2}},
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


@pytest.fixture
def client():
    with patch("app.core.dependencies.get_config", return_value=copy.deepcopy(SAMPLE_CONFIG)):
        with patch("app.core.dependencies.reload_config", return_value=copy.deepcopy(SAMPLE_CONFIG)):
            from app.main import app
            yield TestClient(app)


# ── health ──────────────────────────────────────────

class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── GET /api/v1/admin/districts ─────────────────────

class TestDistricts:
    def test_list_districts(self, client):
        resp = client.get("/api/v1/admin/districts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "pyeongtaek_b"
        assert data[1]["id"] == "busan_bukgu_gap"

    def test_district_has_candidates(self, client):
        resp = client.get("/api/v1/admin/districts")
        data = resp.json()
        assert len(data[0]["candidates"]) == 1
        assert data[0]["candidates"][0]["name"] == "김용남"


# ── GET /api/v1/admin/config ────────────────────────

class TestConfig:
    def test_get_full_config(self, client):
        resp = client.get("/api/v1/admin/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "districts" in data
        assert "rag" in data
        assert "vectordb" in data

    def test_get_rag_config(self, client):
        resp = client.get("/api/v1/admin/config/rag")
        assert resp.status_code == 200
        data = resp.json()
        assert data["retriever"]["top_k"] == 20
        assert data["retriever"]["lookback_days"] == 14
        assert data["purge_days"] == 60


# ── PATCH /api/v1/admin/config/rag ──────────────────

class TestUpdateRAGConfig:
    def test_update_lookback_days(self, client):
        with patch("app.api.v1.routes.admin.save_config"):
            resp = client.patch(
                "/api/v1/admin/config/rag",
                json={"lookback_days": 7},
            )
        assert resp.status_code == 200
        assert resp.json()["retriever"]["lookback_days"] == 7

    def test_update_scorer_provider(self, client):
        with patch("app.api.v1.routes.admin.save_config"):
            resp = client.patch(
                "/api/v1/admin/config/rag",
                json={"scorer_provider": "anthropic", "scorer_model": "claude-sonnet-4-6"},
            )
        assert resp.status_code == 200
        assert resp.json()["scorer"]["provider"] == "anthropic"

    def test_update_multiple_fields(self, client):
        with patch("app.api.v1.routes.admin.save_config"):
            resp = client.patch(
                "/api/v1/admin/config/rag",
                json={"top_k": 10, "min_score": 0.5, "purge_days": 30},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retriever"]["top_k"] == 10
        assert data["reranker"]["min_score"] == 0.5
        assert data["purge_days"] == 30

    def test_invalid_provider_rejected(self, client):
        resp = client.patch(
            "/api/v1/admin/config/rag",
            json={"scorer_provider": "invalid"},
        )
        assert resp.status_code == 422


# ── GET /api/v1/admin/vectordb/stats ────────────────

class TestVectorDBStats:
    def test_stats(self, client):
        mock_repo = MagicMock()
        mock_repo.count.return_value = 207
        with patch("app.api.v1.routes.admin.get_vector_repo", return_value=mock_repo):
            resp = client.get("/api/v1/admin/vectordb/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "chroma"
        assert data["count"] == 207


# ── POST /api/v1/admin/vectordb/purge ───────────────

class TestVectorDBPurge:
    def test_purge(self, client):
        mock_repo = MagicMock()
        mock_repo.delete_older_than.return_value = 5
        mock_repo.count.return_value = 202
        with patch("app.api.v1.routes.admin.get_vector_repo", return_value=mock_repo):
            resp = client.post(
                "/api/v1/admin/vectordb/purge",
                json={"purge_days": 30},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 5
        assert data["remaining"] == 202

    def test_purge_invalid_days(self, client):
        resp = client.post(
            "/api/v1/admin/vectordb/purge",
            json={"purge_days": 0},
        )
        assert resp.status_code == 422


# ── POST /api/v1/admin/pipeline/run ─────────────────

class TestPipelineRun:
    def test_run_pipeline(self, client):
        with patch.object(PipelineRunner, "run_pipeline") as mock_run:
            mock_run.return_value = TaskState(task_id="abc123", status=TaskStatus.RUNNING)
            resp = client.post(
                "/api/v1/admin/pipeline/run",
                json={"scraper": "naver", "days": 3},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "abc123"
        assert data["status"] == "running"

    def test_run_pipeline_conflict(self, client):
        with patch.object(PipelineRunner, "run_pipeline", side_effect=RuntimeError("이미 실행 중")):
            resp = client.post(
                "/api/v1/admin/pipeline/run",
                json={},
            )
        assert resp.status_code == 409

    def test_invalid_scraper_rejected(self, client):
        resp = client.post(
            "/api/v1/admin/pipeline/run",
            json={"scraper": "invalid"},
        )
        assert resp.status_code == 422


# ── GET /api/v1/admin/pipeline/status ───────────────

class TestPipelineStatus:
    def test_idle(self, client):
        with patch.object(PipelineRunner, "current_task", new_callable=lambda: property(lambda self: None)):
            resp = client.get("/api/v1/admin/pipeline/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_running(self, client):
        from datetime import datetime
        task = TaskState(task_id="xyz", status=TaskStatus.RUNNING, started_at=datetime(2026, 5, 1))
        with patch.object(PipelineRunner, "current_task", new_callable=lambda: property(lambda self: task)):
            resp = client.get("/api/v1/admin/pipeline/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["task_id"] == "xyz"


# ── POST /api/v1/admin/pipeline/rebuild ─────────────

class TestPipelineRebuild:
    def test_rebuild(self, client):
        with patch.object(PipelineRunner, "run_rebuild") as mock_rebuild:
            mock_rebuild.return_value = TaskState(task_id="rebuild1", status=TaskStatus.RUNNING)
            resp = client.post("/api/v1/admin/pipeline/rebuild")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "rebuild1"
        assert "재구축" in data["message"]


# ── PipelineRunner 단위 테스트 ──────────────────────

class TestPipelineRunner:
    def test_initial_state_idle(self):
        runner = PipelineRunner()
        assert not runner.is_running
        assert runner.current_task is None

    def test_cannot_run_twice(self):
        runner = PipelineRunner()
        runner._current = TaskState(task_id="t1", status=TaskStatus.RUNNING)
        with pytest.raises(RuntimeError, match="이미 실행 중"):
            runner.run_pipeline({})
