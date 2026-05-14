from __future__ import annotations

import logging
import re

from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(
    r"(?<=[다요죠음됨함임까니])\.\s+"
    r"|(?<=\.)$"
)


def _split_lead(text: str, max_chars: int) -> tuple[str, str]:
    """첫 문단(또는 max_chars 이내)을 리드로 분리. (lead, body) 반환."""
    parts = text.split("\n\n", 1)
    lead = parts[0].strip()

    if len(lead) <= max_chars:
        body = parts[1].strip() if len(parts) > 1 else ""
        return lead, body

    cut = lead[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > max_chars // 3:
        lead = cut[: last_period + 1].strip()
    else:
        last_newline = cut.rfind("\n")
        if last_newline > max_chars // 3:
            lead = cut[:last_newline].strip()
        else:
            lead = cut.strip()

    body = text[len(lead) :].strip()
    return lead, body


@ChunkerRegistry.register("article_aware")
class ArticleAwareChunker(AbstractChunker):
    """기사 구조(제목·리드·본문)를 인식하여 청킹.

    - 첫 번째 청크: [제목] + 리드(첫 문단)
    - 이후 청크: [제목] + 본문 구간 (문단 경계 존중, overlap 이월)
    - 모든 청크에 제목이 prefix로 포함되어 VectorDB 검색 시 맥락 유지
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        lead_max_chars: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lead_max_chars = lead_max_chars
        self._loaded = False

    @property
    def name(self) -> str:
        return "article_aware"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        logger.info("[%s] loaded (no external deps)", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        title = metadata.get("title", "")
        title_prefix = f"[제목] {title}\n" if title else ""
        prefix_len = len(title_prefix)

        lead, body = _split_lead(text, self.lead_max_chars)

        chunks: list[Chunk] = []

        first_text = f"{title_prefix}[리드] {lead}" if lead else title_prefix.strip()
        chunks.append(self._make_chunk(first_text, metadata, 0))

        if not body:
            return chunks

        effective_size = max(self.chunk_size - prefix_len, 100)

        separators = body.split("\n\n")
        paragraphs = [p.strip() for p in separators if p.strip()]
        if len(paragraphs) <= 1 and body.count("\n") > 1:
            paragraphs = [p.strip() for p in body.split("\n") if p.strip()]

        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) + 2 <= effective_size:
                buffer += ("\n\n" + para) if buffer else para
            else:
                if buffer:
                    chunk_text = f"{title_prefix}{buffer}"
                    chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))
                tail = buffer[-self.overlap :] if len(buffer) > self.overlap else buffer
                buffer = (tail + "\n\n" + para) if tail else para

        if buffer:
            chunk_text = f"{title_prefix}{buffer}"
            chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))

        return chunks
