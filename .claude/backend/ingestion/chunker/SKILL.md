---
name: chunker
description: >
  Election Radar 프로젝트의 텍스트 청킹 컴포넌트를 구현하거나 수정할 때 사용.
  KoreanParagraphChunker, SentenceChunker, TokenChunker, SemanticChunker,
  RecursiveChunker 추가·변경, AbstractChunker 인터페이스 수정, ChunkerRegistry
  등록, 청킹 설정(config.yaml) 변경, 청커 테스트 작성 시 반드시 이 파일을 먼저 읽으세요.
  Scraper에서 메모리로 전달된 list[RawArticle]을 입력으로 받습니다.
---

# Chunker 컴포넌트 가이드

## 역할과 데이터 흐름

Scraper가 수집한 `list[RawArticle]`을 메모리에서 직접 전달받아
`list[Chunk]`로 변환합니다. 파일 I/O 없이 메모리 내에서만 동작합니다.

```
Scraper.scrape() → list[RawArticle]   (메모리 전달, 파일 저장 없음)
        ↓
AbstractChunker.chunk(text, metadata) → list[Chunk]
        ↓
Embedder.embed(list[Chunk]) → list[ChunkWithEmbedding]
```

**청커는 텍스트 분할만** 담당합니다.
임베딩, 벡터 저장, 후보 판별은 이후 단계가 처리합니다.

---

## 파일 구조

```
ingestion/chunker/
├── base.py                  ← AbstractChunker ABC + ChunkerRegistry (반드시 먼저 읽기)
├── korean_paragraph.py      ← KoreanParagraphChunker (문단 + 오버랩)
├── sentence.py              ← SentenceChunker (kss 문장 분리)
├── token.py                 ← TokenChunker (tiktoken 토큰 기준)
├── semantic.py              ← SemanticChunker (임베딩 유사도 경계 감지)
├── recursive.py             ← RecursiveChunker (재귀적 계층 분리)
└── __init__.py              ← 5개 구현체 import → Registry 자동 등록
```

> **`__init__.py` 필수**: 새 청커 추가 시 반드시 import 추가.
> import가 없으면 `@ChunkerRegistry.register` 데코레이터가 실행되지 않아 등록 누락.

```python
# ingestion/chunker/__init__.py
from ingestion.chunker import korean_paragraph  # KoreanParagraphChunker 등록
from ingestion.chunker import sentence          # SentenceChunker 등록
from ingestion.chunker import token             # TokenChunker 등록
from ingestion.chunker import semantic          # SemanticChunker 등록
from ingestion.chunker import recursive         # RecursiveChunker 등록
```

---

## 도메인 모델

```python
# models/chunk.py

from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Chunk(BaseModel):
    """청커 출력 단위. 임베딩 이전 상태."""

    # 텍스트
    text: str
    chunk_index: int            # 동일 기사 내 순번 (0-based)
    char_count: int

    # 원본 기사 역참조 (metadata로 VectorDB에 저장됨)
    article_url: str
    source: str                 # scraper.source_name
    title: str                  # 원본 기사 제목
    published_at: datetime
    candidate: str              # pipeline이 주입
    district_id: str            # pipeline이 주입

    # 청커 메타
    chunker_type: str           # Registry 키 ("korean_paragraph" 등)


class ChunkWithEmbedding(Chunk):
    """임베더 출력 단위. VectorDB 저장 직전 상태."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float]

    @property
    def metadata(self) -> dict:
        """VectorDB payload용 직렬화. embedding 제외."""
        return self.model_dump(exclude={"id", "embedding"})
```

---

## AbstractChunker 인터페이스 (Template Method 패턴)

`chunk()`는 부모 클래스의 **concrete 메서드**로, 공통 로깅·빈 텍스트 체크·자동 로드를 처리합니다.
구현체는 `_do_chunk()`만 오버라이드합니다.

```python
# ingestion/chunker/base.py

import logging
from abc import ABC, abstractmethod
from ingestion.base_registry import ComponentRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


class AbstractChunker(ABC):

    @abstractmethod
    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """구현체가 오버라이드. 순수 분할 로직만 담당."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """
        공통 진입점 (Template Method).
        빈 텍스트 체크, 자동 load(), 로깅을 처리한 뒤 _do_chunk() 호출.
        구현체는 이 메서드를 오버라이드하지 않는다.
        """
        if not text.strip():
            logger.warning("[%s] 빈 텍스트 입력 — 스킵", self.name)
            return []
        if not self.is_loaded:
            self.load()

        title = metadata.get("title", "")
        logger.info("[%s] 청킹 시작 — 입력 %d자, 제목='%s'", self.name, len(text), title[:30])

        chunks = self._do_chunk(text, metadata)

        sizes = [c.char_count for c in chunks]
        logger.info(
            "[%s] 청킹 완료 — %d개 청크 생성, 평균 %d자, 최소 %d자, 최대 %d자",
            self.name, len(chunks),
            sum(sizes) // len(sizes) if sizes else 0,
            min(sizes) if sizes else 0,
            max(sizes) if sizes else 0,
        )
        for c in chunks:
            logger.debug("[%s] chunk[%d] %d자: '%s...'", self.name, c.chunk_index, c.char_count, c.text[:40])

        return chunks

    def _make_chunk(self, text: str, metadata: dict, idx: int) -> Chunk:
        return Chunk(
            text=text.strip(),
            chunk_index=idx,
            char_count=len(text.strip()),
            chunker_type=self.name,
            **metadata,
        )


ChunkerRegistry = ComponentRegistry(AbstractChunker, "Chunker")
```

### chunk() vs _do_chunk() 호출 흐름

```
외부 호출 → chunk(text, metadata)         ← AbstractChunker (공통)
               │
               ├── 빈 텍스트 체크 → WARNING 로그 + return []
               ├── is_loaded 체크 → 미로드 시 자동 load()
               ├── INFO 로그: 청킹 시작
               │
               └── _do_chunk(text, metadata)   ← 구현체 (오버라이드)
                      │
                      └── list[Chunk] 반환
               │
               ├── INFO 로그: 청킹 완료 (통계)
               └── DEBUG 로그: 개별 청크 미리보기
```

> **구현체 주의사항**: `_do_chunk()`에서는 빈 텍스트 체크, 자동 load(), 로깅을 하지 않습니다.
> 이 로직은 모두 부모 `chunk()`에서 처리됩니다.

---

## 청커별 상세 명세

### 1. KoreanParagraphChunker `korean_paragraph`

**원리**: `\n\n` 기준 문단 분리 → 목표 크기 초과 시 슬라이딩 오버랩으로 새 청크 시작.
외부 라이브러리 없음. **가장 가볍고 빠름. 기본값으로 권장.**

```python
# ingestion/chunker/korean_paragraph.py

import logging
from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("korean_paragraph")
class KoreanParagraphChunker(AbstractChunker):

    def __init__(self, chunk_size: int = 400, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._loaded = False

    @property
    def name(self) -> str:
        return "korean_paragraph"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        logger.info("[%s] loaded (no external deps)", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        buffer = ""

        for para in paragraphs:
            if len(buffer) + len(para) <= self.chunk_size:
                buffer += ("\n\n" + para) if buffer else para
            else:
                if buffer:
                    chunks.append(self._make_chunk(buffer, metadata, len(chunks)))
                tail = buffer[-self.overlap:] if len(buffer) > self.overlap else buffer
                buffer = (tail + "\n\n" + para) if tail else para

        if buffer:
            chunks.append(self._make_chunk(buffer, metadata, len(chunks)))

        return chunks
```

---

### 2. SentenceChunker `sentence`

**원리**: `kss`로 문장 단위 분리 → N문장씩 그룹화.
문장 경계를 정확히 지킴. 짧은 뉴스 기사에 적합.

**Lazy Load 대상**: `kss` (한국어 문장 분리 라이브러리)

```python
# ingestion/chunker/sentence.py

import logging
from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("sentence")
class SentenceChunker(AbstractChunker):

    def __init__(self, sentences_per_chunk: int = 5) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self._kss = None          # lazy load 대상
        self._loaded = False

    @property
    def name(self) -> str:
        return "sentence"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import kss
        self._kss = kss
        self._loaded = True
        logger.info("[%s] kss loaded", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        sentences: list[str] = self._kss.split_sentences(text)
        chunks: list[Chunk] = []

        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = sentences[i: i + self.sentences_per_chunk]
            chunks.append(self._make_chunk(" ".join(group), metadata, len(chunks)))

        return chunks
```

---

### 3. TokenChunker `token`

**원리**: `tiktoken`으로 토큰 수 기준 분리.
LLM 컨텍스트 윈도우를 정확히 맞춰야 할 때 사용.

**Lazy Load 대상**: `tiktoken`

```python
# ingestion/chunker/token.py

import logging
from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("token")
class TokenChunker(AbstractChunker):

    def __init__(
        self,
        tokens_per_chunk: int = 256,
        overlap_tokens: int = 32,
        encoding_name: str = "cl100k_base",   # GPT-4 / Claude 호환
    ) -> None:
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name
        self._enc = None          # lazy load 대상
        self._loaded = False

    @property
    def name(self) -> str:
        return "token"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import tiktoken
        self._enc = tiktoken.get_encoding(self.encoding_name)
        self._loaded = True
        logger.info("[%s] tiktoken(%s) loaded", self.name, self.encoding_name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        token_ids: list[int] = self._enc.encode(text)
        chunks: list[Chunk] = []
        start = 0

        while start < len(token_ids):
            end = min(start + self.tokens_per_chunk, len(token_ids))
            window = token_ids[start:end]
            chunk_text = self._enc.decode(window)
            chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))
            if end == len(token_ids):
                break
            start += self.tokens_per_chunk - self.overlap_tokens  # 오버랩 적용

        return chunks
```

---

### 4. SemanticChunker `semantic`

**원리**: 연속 문장 쌍의 임베딩 코사인 유사도를 계산 →
유사도가 급격히 낮아지는 지점(주제 전환)을 경계로 분리.
**주제 전환이 많은 긴 기사에 최적. 가장 무거운 청커.**

**Lazy Load 대상**: `sentence_transformers` (SentenceTransformer 모델)

```python
# ingestion/chunker/semantic.py

import logging
import numpy as np
from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)


@ChunkerRegistry.register("semantic")
class SemanticChunker(AbstractChunker):

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        breakpoint_threshold: float = 0.3,  # 유사도가 이 값 이하면 경계
        min_chunk_size: int = 100,           # 최소 청크 크기(자) — 너무 작은 청크 방지
    ) -> None:
        self.model_name = model_name
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self._model = None        # lazy load 대상
        self._loaded = False

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("[%s] loading model: %s ...", self.name, self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._loaded = True
        logger.info("[%s] model loaded", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        import kss
        import numpy as np

        sentences: list[str] = kss.split_sentences(text)
        if len(sentences) <= 1:
            return [self._make_chunk(text, metadata, 0)]

        # 문장 임베딩
        embeddings = self._model.encode(sentences, normalize_embeddings=True)

        # 연속 문장 쌍의 코사인 유사도 계산
        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # 유사도 급락 지점 = 청크 경계
        boundaries: list[int] = [
            i + 1
            for i, sim in enumerate(similarities)
            if sim < self.breakpoint_threshold
        ]

        # 경계 기준으로 문장 그룹 분리
        groups: list[list[str]] = []
        prev = 0
        for boundary in boundaries:
            groups.append(sentences[prev:boundary])
            prev = boundary
        groups.append(sentences[prev:])

        # min_chunk_size 미만 청크는 다음 청크와 병합
        merged: list[str] = []
        buffer = ""
        for group in groups:
            candidate = " ".join(group)
            if len(buffer) + len(candidate) < self.min_chunk_size:
                buffer += (" " + candidate) if buffer else candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = candidate
        if buffer:
            merged.append(buffer)

        return [self._make_chunk(t, metadata, i) for i, t in enumerate(merged)]
```

---

### 5. RecursiveChunker `recursive`

**원리**: 구분자 우선순위(`\n\n` → `\n` → `. ` → ` ` → `""`)를 순서대로 시도.
목표 크기 이하가 될 때까지 재귀적으로 더 작은 구분자로 분리.
LangChain의 `RecursiveCharacterTextSplitter`와 동일한 전략을 직접 구현.

**Lazy Load 대상**: 없음 (순수 Python). load()는 즉시 완료.

```python
# ingestion/chunker/recursive.py

import logging
from ingestion.chunker.base import AbstractChunker, ChunkerRegistry
from models.chunk import Chunk

logger = logging.getLogger(__name__)

# 구분자 우선순위: 큰 단위 → 작은 단위 순
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@ChunkerRegistry.register("recursive")
class RecursiveChunker(AbstractChunker):

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self._loaded = False

    @property
    def name(self) -> str:
        return "recursive"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        logger.info("[%s] loaded (no external deps)", self.name)

    def _do_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        raw_chunks = self._split(text, self.separators)
        # 오버랩 적용
        result: list[str] = []
        for i, c in enumerate(raw_chunks):
            if i == 0 or not result:
                result.append(c)
            else:
                tail = result[-1][-self.overlap:] if len(result[-1]) > self.overlap else result[-1]
                result.append(tail + c)

        return [self._make_chunk(t, metadata, i) for i, t in enumerate(result) if t.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """재귀 분리 핵심 로직."""
        if not separators:
            # 마지막 수단: 문자 단위 강제 분리
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining = separators[1:]

        if len(text) <= self.chunk_size:
            return [text]

        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        buffer = ""

        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                # part 자체가 chunk_size 초과 시 재귀 분리
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, remaining))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            chunks.append(buffer)

        return chunks
```

---

## load() 설계 원칙

모든 청커는 `load()` / `is_loaded` / lazy import 패턴을 동일하게 따릅니다.

```
pipeline.py 시작
    │
    ├── chunker.load()          ← 모델·라이브러리 로드 (1회만)
    │      │
    │      ├── KoreanParagraphChunker: 즉시 완료 (외부 deps 없음)
    │      ├── SentenceChunker:        kss import + 초기화
    │      ├── TokenChunker:           tiktoken encoding 로드
    │      ├── SemanticChunker:        SentenceTransformer 모델 다운로드·로드 (수십 초)
    │      └── RecursiveChunker:       즉시 완료 (외부 deps 없음)
    │
    └── for article in articles:
            chunker.chunk(article.body, metadata)   ← 이미 로드됨, 즉시 실행
```

**규칙**:
- `chunk()` (부모) 내부에서 `is_loaded` 체크 후 미로드 시 자동 `load()` 호출 — 단독 사용 보장
- `load()`는 **멱등(idempotent)**: 여러 번 호출해도 모델을 중복 로드하지 않음
- `SemanticChunker.load()`는 모델 다운로드가 포함되므로 **pipeline 초기화 시 반드시 선행 호출**
- 구현체는 `_do_chunk()`만 오버라이드. `chunk()`를 오버라이드하면 로깅이 누락됨

### 무거운 청커 경고 로그 예시

```python
# pipeline.py에서 SemanticChunker 사용 시 경고
if isinstance(chunker, SemanticChunker):
    logger.warning(
        "SemanticChunker: 모델 로드에 수십 초 소요될 수 있습니다. "
        "초기 실행 시 인터넷 연결이 필요합니다."
    )
chunker.load()
```

---

## 청커별 비교표

| 청커 | Registry 키 | 외부 라이브러리 | load() 비용 | 최적 상황 |
|------|------------|----------------|------------|----------|
| KoreanParagraphChunker | `korean_paragraph` | 없음 | 즉시 | **기본값 권장**, 일반 뉴스 기사 |
| SentenceChunker | `sentence` | `kss` | 낮음 (~1초) | 짧은 기사, 문장 경계 중요 시 |
| TokenChunker | `token` | `tiktoken` | 낮음 (~0.5초) | LLM 토큰 한계 정밀 제어 시 |
| SemanticChunker | `semantic` | `sentence-transformers` | 높음 (~30초+) | 긴 기사, 주제 전환 많을 때 |
| RecursiveChunker | `recursive` | 없음 | 즉시 | 구분자 전략 커스터마이징 필요 시 |

---

## config.yaml 설정 예시

```yaml
chunker:
  type: korean_paragraph       # 5개 중 Registry 키로 선택
  params:
    chunk_size: 400
    overlap: 50

# SemanticChunker 전환 시:
# chunker:
#   type: semantic
#   params:
#     model_name: "BAAI/bge-m3"
#     breakpoint_threshold: 0.3
#     min_chunk_size: 100

# RecursiveChunker 커스텀 구분자:
# chunker:
#   type: recursive
#   params:
#     chunk_size: 400
#     overlap: 50
#     separators: ["\n\n", "\n", ". ", " "]
```

---

## pipeline.py 연동 방식

```python
# ingestion/pipeline.py (청커 관련 부분)

from ingestion.chunker.base import ChunkerRegistry
from models.article import RawArticle
from models.chunk import Chunk

class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        # Registry에서 config 기반으로 청커 생성
        self._chunker = ChunkerRegistry.create(
            config.chunker.type,
            **config.chunker.params
        )
        # 파이프라인 시작 시 선행 로드 (SemanticChunker 대비)
        self._chunker.load()

    def _chunk_articles(self, articles: list[RawArticle]) -> list[Chunk]:
        """
        Scraper 메모리 전달 → Chunker 처리.
        파일 저장 없이 메모리 내에서만 동작.
        """
        all_chunks: list[Chunk] = []

        for article in articles:
            metadata = {
                "article_url":  article.url,
                "source":       article.source,
                "title":        article.title,
                "published_at": article.published_at,
                "candidate":    article.candidate,
                "district_id":  article.district_id,
            }
            chunks = self._chunker.chunk(article.body, metadata)
            all_chunks.extend(chunks)

        return all_chunks
```

---

## 테스트 작성 가이드

외부 라이브러리(kss, tiktoken, sentence-transformers)는 mock 또는 실제 import 중
상황에 맞게 선택합니다.

```python
# tests/ingestion/test_chunker.py

import pytest
from datetime import datetime
from models.article import RawArticle

SAMPLE_TEXT = """
첫 번째 문단입니다. 이 문단은 후보 A에 대한 내용을 담고 있습니다.
평택을 선거구에서 여론조사 결과가 발표되었습니다.

두 번째 문단입니다. 지지율 변동이 감지되었습니다.
후보 B와의 격차가 줄어들고 있다는 분석이 나왔습니다.

세 번째 문단입니다. 전문가들은 막판 변수를 주시하고 있습니다.
"""

SAMPLE_METADATA = {
    "article_url":  "https://example.com/article/1",
    "source":       "naver_news",
    "title":        "평택을 판세 분석",
    "published_at": datetime(2026, 5, 1, 9, 0),
    "candidate":    "후보A",
    "district_id":  "pyeongtaek_b",
}


# ── KoreanParagraphChunker ──────────────────────────────────

def test_korean_paragraph_basic():
    from ingestion.chunker.korean_paragraph import KoreanParagraphChunker
    chunker = KoreanParagraphChunker(chunk_size=200, overlap=30)
    chunker.load()

    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)

    assert len(chunks) >= 1
    assert all(c.char_count <= 250 for c in chunks)   # overlap 감안
    assert all(c.chunker_type == "korean_paragraph" for c in chunks)
    assert all(c.candidate == "후보A" for c in chunks)


def test_korean_paragraph_empty_text():
    from ingestion.chunker.korean_paragraph import KoreanParagraphChunker
    chunker = KoreanParagraphChunker()
    assert chunker.chunk("", SAMPLE_METADATA) == []
    assert chunker.chunk("   ", SAMPLE_METADATA) == []


# ── SentenceChunker ────────────────────────────────────────

def test_sentence_chunker_load():
    from ingestion.chunker.sentence import SentenceChunker
    chunker = SentenceChunker(sentences_per_chunk=3)
    assert not chunker.is_loaded
    chunker.load()
    assert chunker.is_loaded
    assert chunker._kss is not None


def test_sentence_chunker_chunk_index_sequential():
    from ingestion.chunker.sentence import SentenceChunker
    chunker = SentenceChunker(sentences_per_chunk=2)
    chunker.load()
    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# ── TokenChunker ───────────────────────────────────────────

def test_token_chunker_respects_token_limit():
    from ingestion.chunker.token import TokenChunker
    chunker = TokenChunker(tokens_per_chunk=50, overlap_tokens=10)
    chunker.load()
    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    # 각 청크의 토큰 수가 제한을 지키는지 검증
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        assert len(enc.encode(c.text)) <= 60   # 오버랩 감안 여유


# ── SemanticChunker ────────────────────────────────────────

def test_semantic_chunker_load_sets_model():
    """모델 로드 후 _model이 None이 아닌지 확인."""
    from ingestion.chunker.semantic import SemanticChunker
    chunker = SemanticChunker(model_name="BAAI/bge-m3")
    assert not chunker.is_loaded
    chunker.load()   # 실제 모델 로드 — CI에서는 skip 권장
    assert chunker.is_loaded
    assert chunker._model is not None


@pytest.mark.skip(reason="모델 다운로드 필요 — 로컬에서만 실행")
def test_semantic_chunker_splits_on_topic_change():
    from ingestion.chunker.semantic import SemanticChunker
    chunker = SemanticChunker(breakpoint_threshold=0.5)
    chunker.load()
    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    assert len(chunks) >= 1


# ── RecursiveChunker ───────────────────────────────────────

def test_recursive_chunker_custom_separators():
    from ingestion.chunker.recursive import RecursiveChunker
    chunker = RecursiveChunker(chunk_size=150, overlap=20, separators=["\n\n", "\n"])
    chunker.load()
    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    assert len(chunks) >= 1
    assert all(len(c.text) <= 180 for c in chunks)   # overlap 감안


def test_recursive_chunker_idempotent_load():
    from ingestion.chunker.recursive import RecursiveChunker
    chunker = RecursiveChunker()
    chunker.load()
    chunker.load()   # 2회 호출해도 문제없어야 함
    assert chunker.is_loaded


# ── Registry ───────────────────────────────────────────────

def test_chunker_registry_all_registered():
    import ingestion.chunker   # __init__.py 실행 → 5개 등록
    from ingestion.chunker.base import ChunkerRegistry
    available = ChunkerRegistry.available
    assert "korean_paragraph" in available
    assert "sentence"         in available
    assert "token"            in available
    assert "semantic"         in available
    assert "recursive"        in available


def test_chunker_registry_create_by_name():
    import ingestion.chunker
    from ingestion.chunker.base import ChunkerRegistry
    chunker = ChunkerRegistry.create("korean_paragraph", chunk_size=300, overlap=40)
    assert chunker.name == "korean_paragraph"
```

---

## 로깅 규칙

### 공통 로깅 (AbstractChunker.chunk() — 자동 처리)

| 레벨 | 내용 |
|------|------|
| `WARNING` | 빈 텍스트 입력 시 스킵 |
| `INFO` | 청킹 시작 (입력 글자수, 기사 제목), 청킹 완료 (청크 수, 평균/최소/최대 글자수) |
| `DEBUG` | 개별 청크별 인덱스, 글자수, 텍스트 미리보기 (40자) |

### 구현체별 로깅 (각 청커 load() 등)

| 레벨 | 내용 |
|------|------|
| `INFO` | load() 완료 (라이브러리/모델 로드 상태) |

> **구현체별 세부 로그 추가 대비**: 모든 구현체에 `logger = logging.getLogger(__name__)` 선언 유지.
> 향후 구현체 고유의 디버깅 로그가 필요할 때 바로 사용 가능.

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| 구현체에서 `chunk()` 오버라이드 | `_do_chunk()`를 오버라이드. `chunk()`는 부모의 공통 메서드 |
| `_do_chunk()`에서 빈 텍스트 체크/자동 load | 부모 `chunk()`가 처리. `_do_chunk()`는 순수 분할 로직만 |
| 모듈 최상단에서 `import kss` | `load()` 내에서 lazy import |
| `__init__.py`에 새 청커 import 누락 | 파일 추가 즉시 `__init__.py` 업데이트 |
| `chunk_index` 수동 관리 오류 | `_make_chunk()`의 `len(chunks)` 패턴 통일 사용 |
| metadata 키 이름이 Chunk 모델 필드와 불일치 | `Chunk` 모델 필드명과 반드시 일치시킬 것 |
| SemanticChunker를 CI 테스트에서 실행 | `@pytest.mark.skip` 처리, 로컬 전용 표시 |
| 구현체에서 `logger` 선언 제거 | 향후 세부 로그 추가 대비로 유지 |

---

## 관련 파일 참조

- 전체 아키텍처: `CLAUDE.md`
- 범용 Registry 구현: `backend/ingestion/base_registry.py`
- 도메인 모델: `backend/models/chunk.py`
- Scraper 가이드: `.claude/backend/ingestion/scraper/SKILL.md`
- Pipeline Orchestrator: `backend/ingestion/pipeline.py`
