from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "scraped_urls.jsonl"


class ScrapedUrlStore:
    """수집된 기사 URL을 JSONL 파일로 영속 저장하여 중복 수집을 방지한다."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_STORE_PATH
        self._urls: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("URL 저장소 파일 없음 — 새로 생성 예정: %s", self._path)
            return
        count = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._urls.add(record["url"])
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("URL 저장소 로드 완료: %d건 (%s)", count, self._path)

    def contains(self, url: str) -> bool:
        return url in self._urls

    def add(self, url: str, source: str, title: str) -> None:
        if url in self._urls:
            return
        self._urls.add(url)
        record = {
            "url": url,
            "source": source,
            "title": title,
            "scraped_at": datetime.now().isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_batch(self, articles: list[dict[str, str]]) -> int:
        """여러 기사 URL을 한 번에 저장. 새로 추가된 건수를 반환한다."""
        new_count = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            now = datetime.now().isoformat()
            for article in articles:
                url = article["url"]
                if url in self._urls:
                    continue
                self._urls.add(url)
                record = {
                    "url": url,
                    "source": article.get("source", ""),
                    "title": article.get("title", ""),
                    "scraped_at": now,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_count += 1
        return new_count

    @property
    def count(self) -> int:
        return len(self._urls)
