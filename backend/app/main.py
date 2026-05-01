"""Election Radar — FastAPI 진입점."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.dependencies import get_config
    from app.core.scheduler import start_scheduler, stop_scheduler

    config = get_config()
    cron = config.get("schedule", {}).get("cron")
    if cron:
        start_scheduler(cron)
        logger.info("스케줄러 시작 — cron='%s'", cron)
    else:
        logger.info("스케줄러 비활성 — schedule.cron 미설정")

    yield

    stop_scheduler()


app = FastAPI(
    title="Election Radar API",
    description="2026 재보궐선거 판세 분석 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.api.v1.routes.admin import router as admin_router  # noqa: E402
from app.api.v1.routes.scores import router as scores_router  # noqa: E402

app.include_router(admin_router, prefix="/api/v1")
app.include_router(scores_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
