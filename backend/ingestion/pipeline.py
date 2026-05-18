"""수집 파이프라인 오케스트레이터: sync_polls → scrape → chunk → embed → store.

사용법:
    PYTHONPATH=. python -m ingestion.pipeline
    PYTHONPATH=. python -m ingestion.pipeline --scraper naver
    PYTHONPATH=. python -m ingestion.pipeline --days 5
    PYTHONPATH=. python -m ingestion.pipeline --skip-embed
    PYTHONPATH=. python -m ingestion.pipeline --skip-chunk
    PYTHONPATH=. python -m ingestion.pipeline --skip-store
    PYTHONPATH=. python -m ingestion.pipeline --skip-polls     # 여론조사 동기화 생략
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

from models.article import RawArticle

load_dotenv()
from models.chunk import Chunk, ChunkWithEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion.pipeline")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class IngestionPipeline:

    def __init__(self, config: dict) -> None:
        self._config = config
        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def run(
        self,
        *,
        scraper_name: str = "all",
        days: int | None = None,
        skip_chunk: bool = False,
        skip_embed: bool = False,
        skip_store: bool = False,
        skip_polls: bool = False,
    ) -> None:
        poll_articles: list[RawArticle] = []
        if not skip_polls:
            poll_articles = self._sync_polls()

        articles = self._scrape(scraper_name=scraper_name, days=days)
        articles.extend(poll_articles)

        if not articles:
            logger.info("수집된 기사가 없습니다. 파이프라인 종료.")
            return

        articles = self._tag(articles)

        self._save_jsonl(
            items=articles,
            path=DATA_DIR / f"articles_{self._timestamp}.jsonl",
            label="기사",
        )

        if skip_chunk:
            logger.info("--skip-chunk 지정 — 청킹/임베딩 생략")
            return

        chunks = self._chunk(articles)
        chunks = self._filter_chunks(chunks)
        if not chunks:
            logger.info("생성된 청크가 없습니다. 파이프라인 종료.")
            return

        self._save_jsonl(
            items=chunks,
            path=DATA_DIR / f"chunks_{self._timestamp}.jsonl",
            label="청크",
        )

        if skip_embed:
            logger.info("--skip-embed 지정 — 임베딩 생략")
            return

        embedded = self._embed(chunks)
        if not embedded:
            logger.info("임베딩 결과가 없습니다. 파이프라인 종료.")
            return

        self._save_jsonl(
            items=embedded,
            path=DATA_DIR / f"embeddings_{self._timestamp}.jsonl",
            label="임베딩",
        )

        stored = 0
        if skip_store:
            logger.info("--skip-store 지정 — VectorDB 저장 생략")
        else:
            stored = self._store(embedded)

        purge_days = self._config.get("rag", {}).get("purge_days")
        if purge_days and not skip_store:
            self._purge(purge_days)

        logger.info("=" * 50)
        logger.info(
            "파이프라인 완료 — 기사 %d건 → 청크 %d개 → 임베딩 %d개 → 저장 %d개",
            len(articles), len(chunks), len(embedded), stored,
        )
        logger.info("=" * 50)

    # ── polls sync ───────────────────────────────────────

    def _sync_polls(self) -> list[RawArticle]:
        """Google Sheets → polls.jsonl 동기화 + source_url 기사 수집."""
        polls_cfg = self._config.get("polls", {})
        poll_type = polls_cfg.get("type", "jsonl")

        if poll_type != "google_sheets":
            logger.info("여론조사 저장소가 google_sheets가 아닙니다 — 동기화 생략")
            return []

        logger.info("=" * 50)
        logger.info("여론조사 Google Sheets 동기화 시작")
        logger.info("=" * 50)

        from rag.gsheets_poll_store import GoogleSheetsPollStore
        from rag.jsonl_poll_store import JsonlPollStore

        params = polls_cfg.get("params", {})

        try:
            gsheets_store = GoogleSheetsPollStore(**params)
            entries = gsheets_store.load_all()
            metas = gsheets_store.load_meta()
        except Exception as e:
            logger.warning("Google Sheets 연결 실패 — %s", e)
            return []

        jsonl_store = JsonlPollStore()
        jsonl_store.save(entries)
        logger.info("polls.jsonl 동기화 완료 — %d건", len(entries))

        poll_articles = self._scrape_poll_articles(metas)
        return poll_articles

    def _scrape_poll_articles(self, metas) -> list[RawArticle]:
        """polls_meta의 source_url에서 기사를 수집한다."""
        from ingestion.scraper.base import fetch_article_body
        from ingestion.scraper.url_store import ScrapedUrlStore

        url_store = ScrapedUrlStore()

        urls_to_fetch = []
        for meta in metas:
            url = meta.source_url.strip()
            if url and not url_store.contains(url):
                urls_to_fetch.append((url, meta))

        if not urls_to_fetch:
            logger.info("여론조사 기사 — 수집할 신규 URL 없음")
            return []

        logger.info("여론조사 기사 수집 — %d건 URL", len(urls_to_fetch))

        articles: list[RawArticle] = []
        for url, meta in urls_to_fetch:
            body = fetch_article_body(url)
            if not body or len(body) < 50:
                logger.warning("여론조사 기사 본문 추출 실패 — %s", url)
                continue

            title = f"[여론조사] {meta.pollster} {meta.survey_date} {meta.district_name}"
            article = RawArticle(
                url=url,
                source="poll",
                title=title,
                body=body,
                published_at=datetime.combine(meta.survey_date, datetime.min.time()),
                district_id=meta.district_id,
                pollster=meta.pollster,
                poll_survey_date=meta.survey_date.isoformat(),
                sample_size=meta.sample_size,
                margin_of_error=meta.margin_of_error,
            )
            articles.append(article)
            url_store.add(url, source="poll", title=title)

        logger.info("여론조사 기사 수집 완료 — %d건", len(articles))
        return articles

    # ── scrape ───────────────────────────────────────────

    def _scrape(self, *, scraper_name: str, days: int | None) -> list[RawArticle]:
        keywords = _collect_keywords(self._config)
        if not keywords:
            logger.error("config.yaml에 키워드가 없습니다.")
            return []

        today = date.today()
        lookback = days or self._config["scrapers"]["naver"]["params"].get("lookback_days", 2)
        date_from = today - timedelta(days=lookback)
        date_to = today

        logger.info("검색 키워드: %s", keywords)
        logger.info("검색 기간: %s ~ %s (%d일)", date_from, date_to, lookback)

        all_articles: list[RawArticle] = []

        if scraper_name in ("naver", "all"):
            logger.info("=" * 50)
            logger.info("네이버 뉴스 수집 시작")
            logger.info("=" * 50)
            all_articles.extend(self._run_naver(keywords, date_from, date_to))

        if scraper_name in ("naver_election", "all"):
            logger.info("=" * 50)
            logger.info("네이버 선거 페이지 수집 시작")
            logger.info("=" * 50)
            all_articles.extend(self._run_naver_election(keywords, date_from, date_to))

        if scraper_name in ("political", "all"):
            logger.info("=" * 50)
            logger.info("정치 전문 매체 수집 시작")
            logger.info("=" * 50)
            all_articles.extend(self._run_political(keywords, date_from, date_to))

        logger.info("총 수집 기사: %d건", len(all_articles))
        return all_articles

    def _run_naver(self, keywords: list[str], date_from: date, date_to: date) -> list[RawArticle]:
        from ingestion.scraper.naver import NaverNewsScraper

        params = self._config["scrapers"]["naver"]["params"]
        scraper = NaverNewsScraper(
            max_articles_per_run=params.get("max_articles_per_run", 50),
            request_delay_sec=params.get("request_delay_sec", 1.5),
            lookback_days=params.get("lookback_days", 2),
        )
        return scraper.scrape(keywords=keywords, date_from=date_from, date_to=date_to)

    def _run_naver_election(self, keywords: list[str], date_from: date, date_to: date) -> list[RawArticle]:
        from ingestion.scraper.naver_election import NaverElectionScraper

        cfg = self._config.get("scrapers", {}).get("naver_election", {}).get("params", {})
        scraper = NaverElectionScraper(
            max_articles_per_run=cfg.get("max_articles_per_run", 100),
            request_delay_sec=cfg.get("request_delay_sec", 1.5),
            lookback_days=cfg.get("lookback_days", 7),
        )
        return scraper.scrape(keywords=keywords, date_from=date_from, date_to=date_to)

    def _run_political(self, keywords: list[str], date_from: date, date_to: date) -> list[RawArticle]:
        from ingestion.scraper.political import PoliticalNewsScraper

        params = self._config["scrapers"]["political"]["params"]
        scraper = PoliticalNewsScraper(
            urls=params.get("urls", []),
            max_articles_per_run=params.get("max_articles_per_run", 30),
            request_delay_sec=params.get("request_delay_sec", 1.5),
            lookback_days=params.get("lookback_days", 2),
        )
        return scraper.scrape(keywords=keywords, date_from=date_from, date_to=date_to)

    # ── tag ──────────────────────────────────────────────

    def _tag(self, articles: list[RawArticle]) -> list[RawArticle]:
        from ingestion.tagger import tag_articles

        districts = self._config.get("districts", [])
        if not districts:
            logger.warning("config.yaml에 districts가 없습니다. 태깅 생략.")
            return articles

        logger.info("=" * 50)
        logger.info("자동 태깅 시작")
        logger.info("=" * 50)

        return tag_articles(articles, districts)

    # ── chunk ────────────────────────────────────────────

    def _chunk(self, articles: list[RawArticle]) -> list[Chunk]:
        import ingestion.chunker  # noqa: F401
        from ingestion.chunker.base import ChunkerRegistry

        cfg = self._config.get("chunker", {})
        chunker = ChunkerRegistry.create(cfg.get("type", "korean_paragraph"), **cfg.get("params", {}))
        chunker.load()

        logger.info("=" * 50)
        logger.info("청킹 시작 — %s", chunker.name)
        logger.info("=" * 50)

        all_chunks: list[Chunk] = []
        for article in articles:
            metadata = {
                "article_url": article.url,
                "source": article.source,
                "title": article.title,
                "published_at": article.published_at,
                "candidate": article.candidate,
                "district_id": article.district_id,
                "pollster": article.pollster,
                "poll_survey_date": article.poll_survey_date,
                "sample_size": article.sample_size,
                "margin_of_error": article.margin_of_error,
            }
            all_chunks.extend(chunker.chunk(article.body, metadata))

        logger.info("청킹 완료 — 기사 %d건 → 청크 %d개", len(articles), len(all_chunks))
        return all_chunks

    # ── filter ───────────────────────────────────────────

    _MIN_CHUNK_CHARS = 50

    def _filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        before = len(chunks)
        filtered = [
            c for c in chunks
            if c.char_count >= self._MIN_CHUNK_CHARS and c.district_id
        ]
        short = sum(1 for c in chunks if c.char_count < self._MIN_CHUNK_CHARS)
        untagged = sum(1 for c in chunks if not c.district_id)
        removed = before - len(filtered)
        if removed:
            logger.info(
                "청크 필터링 — %d건 제거 (50자 미만 %d건, 미태깅 %d건), %d건 유지",
                removed, short, untagged, len(filtered),
            )
        return filtered

    # ── embed ────────────────────────────────────────────

    def _embed(self, chunks: list[Chunk]) -> list[ChunkWithEmbedding]:
        import ingestion.embedder  # noqa: F401
        from ingestion.embedder.base import EmbedderRegistry

        cfg = self._config.get("embedder", {})
        embedder = EmbedderRegistry.create(cfg.get("type", "openai"), **cfg.get("params", {}))
        embedder.load()

        logger.info("=" * 50)
        logger.info("임베딩 시작 — %s (차원=%d)", embedder.name, embedder.dimensions)
        logger.info("=" * 50)

        embedded = embedder.embed(chunks)

        logger.info("임베딩 완료 — 청크 %d개 → 벡터 %d개", len(chunks), len(embedded))
        return embedded

    # ── store ────────────────────────────────────────────

    def _store(self, embedded: list[ChunkWithEmbedding]) -> int:
        import vectordb  # noqa: F401
        from vectordb.base import VectorRepositoryRegistry

        cfg = self._config.get("vectordb", {})
        repo_type = cfg.get("type", "chroma")
        params = cfg.get("params", {})
        collection = cfg.get("collection", "election_chunks")

        repo = VectorRepositoryRegistry.create(repo_type, collection=collection, **params)
        repo.load()

        logger.info("=" * 50)
        logger.info("VectorDB 저장 시작 — %s (collection=%s)", repo.name, collection)
        logger.info("=" * 50)

        count = repo.upsert(embedded)

        logger.info("VectorDB 저장 완료 — %d개 벡터 저장, 총 %d개", count, repo.count())
        return count

    # ── purge ────────────────────────────────────────────

    def _purge(self, days: int) -> None:
        import vectordb  # noqa: F401
        from vectordb.base import VectorRepositoryRegistry

        cfg = self._config.get("vectordb", {})
        repo_type = cfg.get("type", "chroma")
        params = cfg.get("params", {})
        collection = cfg.get("collection", "election_chunks")

        repo = VectorRepositoryRegistry.create(repo_type, collection=collection, **params)
        repo.load()

        deleted = repo.delete_older_than(days)
        logger.info("만료 정리 — %d일 이전 벡터 %d개 삭제", days, deleted)

    # ── 공통 유틸 ────────────────────────────────────────

    @staticmethod
    def _save_jsonl(items: list, path: Path, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
        logger.info("%s %d건 저장: %s", label, len(items), path)


def _collect_keywords(config: dict) -> list[str]:
    keywords: list[str] = []
    for district in config.get("districts", []):
        for candidate in district.get("candidates", []):
            keywords.extend(candidate.get("keywords", []))
    return keywords


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Election Radar 수집 파이프라인")
    parser.add_argument("--scraper", choices=["naver", "political", "all"], default="all")
    parser.add_argument("--days", type=int, default=None, help="오늘 기준 며칠 전부터 검색")
    parser.add_argument("--skip-chunk", action="store_true", help="청킹·임베딩 생략")
    parser.add_argument("--skip-embed", action="store_true", help="임베딩·저장 생략")
    parser.add_argument("--skip-store", action="store_true", help="VectorDB 저장 생략")
    parser.add_argument("--skip-polls", action="store_true", help="여론조사 동기화 생략")
    args = parser.parse_args()

    config = _load_config()
    pipeline = IngestionPipeline(config)
    pipeline.run(
        scraper_name=args.scraper,
        days=args.days,
        skip_chunk=args.skip_chunk,
        skip_embed=args.skip_embed,
        skip_store=args.skip_store,
        skip_polls=args.skip_polls,
    )


if __name__ == "__main__":
    main()
