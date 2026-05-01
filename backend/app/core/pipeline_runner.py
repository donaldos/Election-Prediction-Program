"""백그라운드 파이프라인 실행 관리."""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict | None = None
    error: str | None = None


class PipelineRunner:

    def __init__(self) -> None:
        self._current: TaskState | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.status == TaskStatus.RUNNING

    @property
    def current_task(self) -> TaskState | None:
        with self._lock:
            return self._current

    def run_pipeline(
        self,
        config: dict,
        *,
        scraper_name: str = "all",
        days: int | None = None,
        skip_chunk: bool = False,
        skip_embed: bool = False,
        skip_store: bool = False,
    ) -> TaskState:
        with self._lock:
            if self._current and self._current.status == TaskStatus.RUNNING:
                raise RuntimeError("파이프라인이 이미 실행 중입니다.")

            task_id = uuid.uuid4().hex[:12]
            self._current = TaskState(task_id=task_id, status=TaskStatus.RUNNING, started_at=datetime.now())

        thread = threading.Thread(
            target=self._execute,
            args=(config,),
            kwargs={
                "scraper_name": scraper_name,
                "days": days,
                "skip_chunk": skip_chunk,
                "skip_embed": skip_embed,
                "skip_store": skip_store,
            },
            daemon=True,
        )
        thread.start()
        return self._current

    def run_rebuild(self, config: dict) -> TaskState:
        with self._lock:
            if self._current and self._current.status == TaskStatus.RUNNING:
                raise RuntimeError("파이프라인이 이미 실행 중입니다.")

            task_id = uuid.uuid4().hex[:12]
            self._current = TaskState(task_id=task_id, status=TaskStatus.RUNNING, started_at=datetime.now())

        thread = threading.Thread(
            target=self._execute_rebuild,
            args=(config,),
            daemon=True,
        )
        thread.start()
        return self._current

    def _execute(self, config: dict, **kwargs) -> None:
        try:
            from ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline(config)
            pipeline.run(**kwargs)

            with self._lock:
                self._current.status = TaskStatus.COMPLETED
                self._current.finished_at = datetime.now()
                self._current.result = {"message": "파이프라인 실행 완료"}
            logger.info("파이프라인 완료 — task_id=%s", self._current.task_id)

        except Exception as e:
            logger.exception("파이프라인 실패 — %s", e)
            with self._lock:
                self._current.status = TaskStatus.FAILED
                self._current.finished_at = datetime.now()
                self._current.error = str(e)

    def _execute_rebuild(self, config: dict) -> None:
        try:
            import shutil
            from pathlib import Path

            cfg = config.get("vectordb", {})
            persist_dir = cfg.get("params", {}).get("persist_dir")

            if persist_dir:
                db_path = Path(__file__).resolve().parent.parent.parent / persist_dir
                if db_path.exists():
                    shutil.rmtree(db_path)
                    logger.info("VectorDB 삭제 완료 — %s", db_path)

            from ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline(config)
            pipeline.run()

            with self._lock:
                self._current.status = TaskStatus.COMPLETED
                self._current.finished_at = datetime.now()
                self._current.result = {"message": "VectorDB 재구축 완료"}
            logger.info("VectorDB 재구축 완료 — task_id=%s", self._current.task_id)

        except Exception as e:
            logger.exception("VectorDB 재구축 실패 — %s", e)
            with self._lock:
                self._current.status = TaskStatus.FAILED
                self._current.finished_at = datetime.now()
                self._current.error = str(e)


pipeline_runner = PipelineRunner()
