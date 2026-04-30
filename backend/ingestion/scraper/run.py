"""스크레이퍼 + 청커 수동 실행 스크립트.

사용법:
    PYTHONPATH=. python -m ingestion.scraper.run
    PYTHONPATH=. python -m ingestion.scraper.run --scraper naver
    PYTHONPATH=. python -m ingestion.scraper.run --scraper political
    PYTHONPATH=. python -m ingestion.scraper.run --days 3
    PYTHONPATH=. python -m ingestion.scraper.run --no-chunk
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scraper.run")

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_keywords(config: dict) -> list[str]:
    keywords: list[str] = []
    for district in config.get("districts", []):
        for candidate in district.get("candidates", []):
            keywords.extend(candidate.get("keywords", []))
    return keywords


def run_naver(config: dict, keywords: list[str], date_from: date, date_to: date) -> list:
    from ingestion.scraper.naver import NaverNewsScraper

    params = config["scrapers"]["naver"]["params"]
    scraper = NaverNewsScraper(
        max_articles_per_run=params.get("max_articles_per_run", 50),
        request_delay_sec=params.get("request_delay_sec", 1.5),
        lookback_days=params.get("lookback_days", 2),
    )
    return scraper.scrape(keywords=keywords, date_from=date_from, date_to=date_to)


def run_political(config: dict, keywords: list[str], date_from: date, date_to: date) -> list:
    from ingestion.scraper.political import PoliticalNewsScraper

    params = config["scrapers"]["political"]["params"]
    scraper = PoliticalNewsScraper(
        urls=params.get("urls", []),
        max_articles_per_run=params.get("max_articles_per_run", 30),
        request_delay_sec=params.get("request_delay_sec", 1.5),
        lookback_days=params.get("lookback_days", 2),
    )
    return scraper.scrape(keywords=keywords, date_from=date_from, date_to=date_to)


def save_articles(articles: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article.model_dump(mode="json"), ensure_ascii=False) + "\n")
    logger.info("기사 %d건 저장 완료: %s", len(articles), output_path)


def run_chunker(config: dict, articles: list) -> list:
    from ingestion.chunker.base import ChunkerRegistry
    import ingestion.chunker  # noqa: F401 — Registry 자동 등록

    chunker_cfg = config.get("chunker", {})
    chunker_type = chunker_cfg.get("type", "korean_paragraph")
    chunker_params = chunker_cfg.get("params", {})

    chunker = ChunkerRegistry.create(chunker_type, **chunker_params)
    chunker.load()

    all_chunks = []
    for article in articles:
        metadata = {
            "article_url": article.url,
            "source": article.source,
            "title": article.title,
            "published_at": article.published_at,
            "candidate": article.candidate,
            "district_id": article.district_id,
        }
        chunks = chunker.chunk(article.body, metadata)
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
    logger.info("청크 %d건 저장 완료: %s", len(chunks), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Election Radar 스크레이퍼 수동 실행")
    parser.add_argument("--scraper", choices=["naver", "political", "all"], default="all", help="실행할 스크레이퍼 (기본: all)")
    parser.add_argument("--days", type=int, default=None, help="오늘 기준 며칠 전부터 검색 (기본: config.yaml lookback_days)")
    parser.add_argument("--no-chunk", action="store_true", help="청킹 단계 생략 (스크레이핑만 실행)")
    args = parser.parse_args()

    config = load_config()
    keywords = collect_keywords(config)

    if not keywords:
        logger.error("config.yaml에 키워드가 없습니다. districts.candidates.keywords를 확인하세요.")
        sys.exit(1)

    logger.info("검색 키워드: %s", keywords)

    today = date.today()
    lookback = args.days or config["scrapers"]["naver"]["params"].get("lookback_days", 2)
    date_from = today - timedelta(days=lookback)
    date_to = today

    logger.info("검색 기간: %s ~ %s (%d일)", date_from, date_to, lookback)

    all_articles = []

    if args.scraper in ("naver", "all"):
        logger.info("=" * 50)
        logger.info("네이버 뉴스 수집 시작")
        logger.info("=" * 50)
        naver_articles = run_naver(config, keywords, date_from, date_to)
        all_articles.extend(naver_articles)

    if args.scraper in ("political", "all"):
        logger.info("=" * 50)
        logger.info("정치 전문 매체 수집 시작")
        logger.info("=" * 50)
        political_articles = run_political(config, keywords, date_from, date_to)
        all_articles.extend(political_articles)

    logger.info("=" * 50)
    logger.info("수집 결과 요약")
    logger.info("=" * 50)
    logger.info("총 수집 기사: %d건", len(all_articles))

    if all_articles:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = DATA_DIR / f"articles_{timestamp}.jsonl"
        save_articles(all_articles, output_path)

        logger.info("-" * 50)
        for i, article in enumerate(all_articles[:10], 1):
            logger.info(
                "[%d] %s | %s | %s",
                i, article.source, article.title[:50], article.url,
            )
        if len(all_articles) > 10:
            logger.info("... 외 %d건", len(all_articles) - 10)

        if not args.no_chunk:
            logger.info("=" * 50)
            logger.info("청킹 시작")
            logger.info("=" * 50)

            all_chunks = run_chunker(config, all_articles)

            logger.info("=" * 50)
            logger.info("청킹 결과 요약")
            logger.info("=" * 50)
            logger.info("총 청크: %d개 (기사 %d건 → 청크 %d개)", len(all_chunks), len(all_articles), len(all_chunks))

            if all_chunks:
                chunks_path = DATA_DIR / f"chunks_{timestamp}.jsonl"
                save_chunks(all_chunks, chunks_path)

                logger.info("-" * 50)
                for i, chunk in enumerate(all_chunks[:5], 1):
                    logger.info(
                        "[%d] chunk[%d] %d자 | %s | '%s...'",
                        i, chunk.chunk_index, chunk.char_count, chunk.title[:30], chunk.text[:50],
                    )
                if len(all_chunks) > 5:
                    logger.info("... 외 %d개 청크", len(all_chunks) - 5)
    else:
        logger.info("수집된 기사가 없습니다.")


if __name__ == "__main__":
    main()
