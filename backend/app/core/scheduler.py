"""APScheduler — cron 주기 자동 수집 + 판정."""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _scheduled_job() -> None:
    from app.core.dependencies import reload_config

    config = reload_config()
    logger.info("스케줄 작업 시작 — 수집 + 판정")

    try:
        from ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(config)
        pipeline.run()
    except Exception as e:
        logger.exception("수집 파이프라인 실패 — %s", e)

    for district in config.get("districts", []):
        try:
            from rag.pipeline import (
                _build_embedder,
                _build_reranker,
                _build_retriever,
                _build_scorer,
                _build_vector_repo,
            )
            from rag.verdict_store import VerdictStore

            embedder = _build_embedder(config)
            vector_repo = _build_vector_repo(config)
            retriever = _build_retriever(config, embedder, vector_repo)
            reranker = _build_reranker(config)

            all_results = retriever.retrieve_for_district(district)
            query = f"{district['name']} 선거 판세"
            all_results = reranker.rerank(query, all_results)

            scorer = _build_scorer(config)
            verdict = scorer.score(all_results, district)

            store = VerdictStore()
            store.save(verdict)
            logger.info("판정 완료 — %s", district["name"])

        except Exception as e:
            logger.exception("판정 실패 — %s: %s", district["name"], e)

    logger.info("스케줄 작업 완료")


def start_scheduler(cron_expression: str) -> BackgroundScheduler:
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.info("스케줄러 이미 실행 중")
        return _scheduler

    parts = cron_expression.split()
    if len(parts) != 5:
        logger.error("잘못된 cron 표현식: '%s'", cron_expression)
        raise ValueError(f"cron 표현식은 5개 필드 (분 시 일 월 요일): '{cron_expression}'")

    trigger = CronTrigger.from_crontab(cron_expression)

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _scheduled_job,
        trigger=trigger,
        id="election_radar_pipeline",
        name="수집 + 판정 자동 실행",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("스케줄러 시작 — cron='%s'", cron_expression)
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("스케줄러 중지")
    _scheduler = None


def get_scheduler() -> BackgroundScheduler | None:
    return _scheduler
