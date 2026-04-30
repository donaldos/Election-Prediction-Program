---
name: embedder
description: >
  Election Radar 프로젝트의 벡터 임베딩 컴포넌트를 구현하거나 수정할 때 사용.
  OpenAIEmbedder, BGEM3Embedder, KoSimCSEEmbedder 추가·변경,
  AbstractEmbedder 인터페이스 수정, EmbedderRegistry 등록,
  임베딩 설정(config.yaml) 변경, 임베더 테스트 작성 시 반드시 이 파일을 먼저 읽으세요.
  Chunker에서 메모리로 전달된 list[Chunk]를 입력으로 받습니다.
---

# Embedder 컴포넌트 가이드

## 역할과 데이터 흐름

Chunker가 분할한 `list[Chunk]`를 메모리에서 전달받아
`list[ChunkWithEmbedding]`으로 변환합니다.

```
Chunker.chunk() → list[Chunk]
        ↓
AbstractEmbedder.embed(list[Chunk]) → list[ChunkWithEmbedding]
        ↓
VectorDB.upsert(list[ChunkWithEmbedding])
```

**임베더는 벡터 변환만** 담당합니다.
텍스트 분할은 Chunker, 벡터 저장은 VectorDB가 처리합니다.

---

## 파일 구조

```
ingestion/embedder/
├── base.py              ← AbstractEmbedder ABC + EmbedderRegistry (반드시 먼저 읽기)
├── openai_embedder.py   ← OpenAIEmbedder (OpenAI API, 기본값)
├── bge.py               ← BGEM3Embedder (로컬 실행)
├── ko_simcse.py         ← KoSimCSEEmbedder (한국어 특화, 로컬 실행)
└── __init__.py          ← 3개 구현체 import → Registry 자동 등록
```

> **`__init__.py` 필수**: 새 임베더 추가 시 반드시 import 추가.
> import가 없으면 `@EmbedderRegistry.register` 데코레이터가 실행되지 않아 등록 누락.

```python
# ingestion/embedder/__init__.py
from ingestion.embedder import openai_embedder  # OpenAIEmbedder 등록
from ingestion.embedder import bge              # BGEM3Embedder 등록
from ingestion.embedder import ko_simcse        # KoSimCSEEmbedder 등록
```

---

## 도메인 모델

Chunk → ChunkWithEmbedding 변환이 임베더의 핵심 역할입니다.

```python
# models/chunk.py (이미 구현됨)

class Chunk(BaseModel):
    text: str
    chunk_index: int
    char_count: int
    article_url: str
    source: str
    title: str
    published_at: datetime
    candidate: str
    district_id: str
    chunker_type: str


class ChunkWithEmbedding(Chunk):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float]

    @property
    def metadata(self) -> dict:
        return self.model_dump(exclude={"id", "embedding"})
```

---

## AbstractEmbedder 인터페이스 (Template Method 패턴)

Chunker와 동일하게 `embed()`는 부모의 **concrete 메서드**로 공통 로깅을 처리하고,
구현체는 `_do_embed()`만 오버라이드합니다.

```python
# ingestion/embedder/base.py

import logging
from abc import ABC, abstractmethod
from ingestion.base_registry import ComponentRegistry
from models.chunk import Chunk, ChunkWithEmbedding

logger = logging.getLogger(__name__)


class AbstractEmbedder(ABC):

    @abstractmethod
    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        """
        텍스트 리스트를 벡터 리스트로 변환.
        구현체가 오버라이드. 순수 임베딩 로직만 담당.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            각 텍스트에 대응하는 벡터(list[float]) 리스트.
            len(결과) == len(texts) 보장.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """출력 벡터 차원 수. VectorDB 컬렉션 생성 시 사용."""
        ...

    def embed(self, chunks: list[Chunk]) -> list[ChunkWithEmbedding]:
        """
        공통 진입점 (Template Method).
        배치 처리, 로깅을 처리한 뒤 _do_embed() 호출.
        구현체는 이 메서드를 오버라이드하지 않는다.
        """
        if not chunks:
            logger.warning("[%s] 빈 청크 리스트 — 스킵", self.name)
            return []
        if not self.is_loaded:
            self.load()

        logger.info("[%s] 임베딩 시작 — %d개 청크", self.name, len(chunks))

        texts = [c.text for c in chunks]
        vectors = self._do_embed(texts)

        results = []
        for chunk, vector in zip(chunks, vectors):
            results.append(
                ChunkWithEmbedding(
                    **chunk.model_dump(),
                    embedding=vector,
                )
            )

        logger.info(
            "[%s] 임베딩 완료 — %d개 벡터 생성, 차원=%d",
            self.name, len(results), self.dimensions,
        )
        return results


EmbedderRegistry = ComponentRegistry(AbstractEmbedder, "Embedder")
```

### embed() vs _do_embed() 호출 흐름

```
외부 호출 → embed(list[Chunk])              ← AbstractEmbedder (공통)
               │
               ├── 빈 리스트 체크 → WARNING 로그 + return []
               ├── is_loaded 체크 → 미로드 시 자동 load()
               ├── INFO 로그: 임베딩 시작
               │
               └── _do_embed(list[str])        ← 구현체 (오버라이드)
                      │
                      └── list[list[float]] 반환
               │
               ├── Chunk + embedding → ChunkWithEmbedding 조립
               └── INFO 로그: 임베딩 완료 (벡터 수, 차원)
```

---

## 임베더별 상세 명세

### 1. OpenAIEmbedder `openai` (기본값)

**원리**: OpenAI Embeddings API 호출. 서버측 추론이라 로컬 GPU 불필요.
**API 키 필수**: `backend/.env`의 `OPENAI_API_KEY` 환경변수 사용.

**Lazy Load 대상**: `openai` 패키지

```python
# ingestion/embedder/openai_embedder.py

import logging
import os
from ingestion.embedder.base import AbstractEmbedder, EmbedderRegistry

logger = logging.getLogger(__name__)


@EmbedderRegistry.register("openai")
class OpenAIEmbedder(AbstractEmbedder):

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
    ) -> None:
        self._model_name = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "openai"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def load(self) -> None:
        if self._loaded:
            return
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._loaded = True
        logger.info("[%s] client initialized — model=%s, dim=%d", self.name, self._model_name, self._dimensions)

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug("[%s] API 호출 — batch %d~%d", self.name, i, i + len(batch))

            response = self._client.embeddings.create(
                model=self._model_name,
                input=batch,
                dimensions=self._dimensions,
            )
            batch_vectors = [item.embedding for item in response.data]
            all_vectors.extend(batch_vectors)

        return all_vectors
```

#### OpenAI 모델별 사양

| 모델 | 차원 | 비용 (1M 토큰) | 비고 |
|------|------|----------------|------|
| text-embedding-3-small | 1536 | ~$0.02 | **기본값 권장**, 비용 대비 성능 최적 |
| text-embedding-3-large | 3072 | ~$0.13 | 최고 정확도, 비용 높음 |
| text-embedding-ada-002 | 1536 | ~$0.10 | 레거시, dimensions 파라미터 미지원 |

#### API 키 설정

```bash
# backend/.env (git 제외)
OPENAI_API_KEY=sk-proj-...
```

코드에서는 `os.environ["OPENAI_API_KEY"]`로 읽습니다.
`.env` 파일은 `.gitignore`에 포함되어 git에 커밋되지 않습니다.

---

### 2. BGEM3Embedder `bge_m3`

**원리**: BAAI/bge-m3 모델을 로컬에서 실행. API 키 불필요.
**Lazy Load 대상**: `FlagEmbedding` (BGEM3FlagModel)

```python
# ingestion/embedder/bge.py

import logging
from ingestion.embedder.base import AbstractEmbedder, EmbedderRegistry

logger = logging.getLogger(__name__)


@EmbedderRegistry.register("bge_m3")
class BGEM3Embedder(AbstractEmbedder):

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "bge_m3"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimensions(self) -> int:
        return 1024  # bge-m3 고정 차원

    def load(self) -> None:
        if self._loaded:
            return
        from FlagEmbedding import BGEM3FlagModel
        logger.info("[%s] loading model: %s ...", self.name, self._model_name)
        self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
        self._loaded = True
        logger.info("[%s] model loaded", self.name)

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._model.encode(batch)
            vectors = result["dense_vecs"].tolist()
            all_vectors.extend(vectors)

        return all_vectors
```

---

### 3. KoSimCSEEmbedder `ko_simcse`

**원리**: 한국어 특화 SimCSE 모델을 로컬에서 실행. API 키 불필요.
**Lazy Load 대상**: `sentence_transformers` (SentenceTransformer)

```python
# ingestion/embedder/ko_simcse.py

import logging
from ingestion.embedder.base import AbstractEmbedder, EmbedderRegistry

logger = logging.getLogger(__name__)


@EmbedderRegistry.register("ko_simcse")
class KoSimCSEEmbedder(AbstractEmbedder):

    def __init__(
        self,
        model_name: str = "BM-K/KoSimCSE-roberta",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "ko_simcse"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimensions(self) -> int:
        return 768  # KoSimCSE-roberta 고정 차원

    def load(self) -> None:
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("[%s] loading model: %s ...", self.name, self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._loaded = True
        logger.info("[%s] model loaded", self.name)

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors = self._model.encode(batch, normalize_embeddings=True)
            all_vectors.extend(vectors.tolist())

        return all_vectors
```

---

## load() 설계 원칙

Chunker와 동일한 패턴을 따릅니다.

```
pipeline.py 시작
    │
    ├── embedder.load()         ← 모델·클라이언트 초기화 (1회만)
    │      │
    │      ├── OpenAIEmbedder:     OpenAI 클라이언트 생성 (~즉시)
    │      ├── BGEM3Embedder:      FlagEmbedding 모델 로드 (~30초, ~2GB)
    │      └── KoSimCSEEmbedder:   SentenceTransformer 모델 로드 (~10초, ~500MB)
    │
    └── for chunk_batch in batches:
            embedder.embed(chunk_batch)   ← 이미 로드됨, 즉시 실행
```

**규칙**:
- `embed()` (부모) 내부에서 `is_loaded` 체크 후 미로드 시 자동 `load()` 호출
- `load()`는 **멱등(idempotent)**: 여러 번 호출해도 중복 초기화하지 않음
- 구현체는 `_do_embed()`만 오버라이드. `embed()`를 오버라이드하면 로깅이 누락됨

---

## 임베더별 비교표

| 임베더 | Registry 키 | API 키 | 벡터 차원 | 비용 | 최적 상황 |
|--------|------------|--------|----------|------|----------|
| OpenAIEmbedder | `openai` | 필요 | 1536/3072 | 유료 | **기본값**, GPU 없는 환경 |
| BGEM3Embedder | `bge_m3` | 불필요 | 1024 | 무료 | GPU 있는 서버, 대량 처리 |
| KoSimCSEEmbedder | `ko_simcse` | 불필요 | 768 | 무료 | 한국어 특화, 경량 로컬 |

---

## config.yaml 설정 예시

```yaml
# OpenAI (기본값)
embedder:
  type: openai
  params:
    model: text-embedding-3-small
    dimensions: 1536
    batch_size: 100

# BGE-M3 (로컬, GPU 권장)
# embedder:
#   type: bge_m3
#   params:
#     model_name: "BAAI/bge-m3"
#     batch_size: 32

# KoSimCSE (로컬, 경량)
# embedder:
#   type: ko_simcse
#   params:
#     model_name: "BM-K/KoSimCSE-roberta"
#     batch_size: 32
```

---

## 환경변수 및 API 키 관리

```
backend/.env              ← 비밀값 저장 (git 제외)
├── OPENAI_API_KEY=sk-...
```

- `.env` 파일은 `.gitignore`에 포함되어 git에 커밋되지 않음
- 코드에서는 `os.environ["OPENAI_API_KEY"]`로 읽음
- `bge_m3`, `ko_simcse`는 API 키 불필요 (로컬 모델)

---

## pipeline.py 연동 방식

```python
# ingestion/pipeline.py (임베더 관련 부분)

from ingestion.embedder.base import EmbedderRegistry
from models.chunk import Chunk, ChunkWithEmbedding

class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self._embedder = EmbedderRegistry.create(
            config.embedder.type,
            **config.embedder.params
        )
        self._embedder.load()

    def _embed_chunks(self, chunks: list[Chunk]) -> list[ChunkWithEmbedding]:
        return self._embedder.embed(chunks)
```

---

## 로깅 규칙

### 공통 로깅 (AbstractEmbedder.embed() — 자동 처리)

| 레벨 | 내용 |
|------|------|
| `WARNING` | 빈 청크 리스트 입력 시 스킵 |
| `INFO` | 임베딩 시작 (청크 수), 임베딩 완료 (벡터 수, 차원) |

### 구현체별 로깅

| 레벨 | 내용 |
|------|------|
| `INFO` | load() 완료 (모델명, 차원 등) |
| `DEBUG` | 배치별 API 호출/모델 추론 진행 |

> **구현체별 세부 로그 추가 대비**: 모든 구현체에 `logger = logging.getLogger(__name__)` 선언 유지.

---

## 테스트 작성 가이드

OpenAI API 호출은 **반드시 mock 처리**합니다.
로컬 모델(bge_m3, ko_simcse)은 모델 다운로드 필요 시 skip 처리합니다.

```python
# tests/ingestion/test_embedder.py

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from models.chunk import Chunk

SAMPLE_CHUNKS = [
    Chunk(
        text="평택을 선거구 여론조사 결과",
        chunk_index=0,
        char_count=14,
        article_url="https://example.com/1",
        source="naver_news",
        title="판세 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보A",
        district_id="pyeongtaek_b",
        chunker_type="korean_paragraph",
    ),
    Chunk(
        text="부산북구갑 후보 간 격차 분석",
        chunk_index=1,
        char_count=15,
        article_url="https://example.com/2",
        source="naver_news",
        title="격차 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보B",
        district_id="busan_bukgu_gap",
        chunker_type="korean_paragraph",
    ),
]


# ── OpenAIEmbedder ────────────────────────────────────────

def test_openai_embedder_embed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    from ingestion.embedder.openai_embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536, batch_size=100)

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]

    with patch("openai.OpenAI") as MockClient:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        MockClient.return_value = mock_client

        embedder.load()
        results = embedder.embed(SAMPLE_CHUNKS)

    assert len(results) == 2
    assert len(results[0].embedding) == 1536
    assert results[0].article_url == "https://example.com/1"
    assert results[1].candidate == "후보B"


def test_openai_embedder_empty_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    from ingestion.embedder.openai_embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    embedder._loaded = True
    assert embedder.embed([]) == []


def test_openai_embedder_dimensions():
    from ingestion.embedder.openai_embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder(dimensions=3072)
    assert embedder.dimensions == 3072


# ── BGEM3Embedder ─────────────────────────────────────────

@pytest.mark.skip(reason="모델 다운로드 필요 — 로컬에서만 실행")
def test_bge_m3_embedder_load():
    from ingestion.embedder.bge import BGEM3Embedder
    embedder = BGEM3Embedder()
    embedder.load()
    assert embedder.is_loaded
    assert embedder.dimensions == 1024


# ── KoSimCSEEmbedder ──────────────────────────────────────

@pytest.mark.skip(reason="모델 다운로드 필요 — 로컬에서만 실행")
def test_ko_simcse_embedder_load():
    from ingestion.embedder.ko_simcse import KoSimCSEEmbedder
    embedder = KoSimCSEEmbedder()
    embedder.load()
    assert embedder.is_loaded
    assert embedder.dimensions == 768


# ── Registry ──────────────────────────────────────────────

def test_embedder_registry_all_registered():
    import ingestion.embedder
    from ingestion.embedder.base import EmbedderRegistry
    names = EmbedderRegistry.registered_names
    assert "openai" in names
    assert "bge_m3" in names
    assert "ko_simcse" in names


def test_embedder_registry_create_by_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    import ingestion.embedder
    from ingestion.embedder.base import EmbedderRegistry
    embedder = EmbedderRegistry.create("openai", model="text-embedding-3-small", dimensions=1536, batch_size=100)
    assert embedder.name == "openai"
```

---

## 새 임베더 구현 체크리스트

### 1단계: 구현 파일 생성

```python
# ingestion/embedder/my_embedder.py

@EmbedderRegistry.register("my_embedder")
class MyEmbedder(AbstractEmbedder):

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        ...

    def _do_embed(self, texts: list[str]) -> list[list[float]]:
        # 순수 임베딩 로직만. 로깅·빈 입력 체크는 부모 embed()가 처리.
        ...

    @property
    def dimensions(self) -> int:
        return 768  # 모델에 따라 고정값 반환
```

### 2단계: `__init__.py`에 등록

```python
from ingestion.embedder import my_embedder  # ← 추가
```

### 3단계: config.yaml 추가

```yaml
embedder:
  type: my_embedder
  params:
    model_name: "my-org/my-model"
    batch_size: 32
```

### 4단계: 테스트 작성 (`tests/ingestion/test_embedder.py` 참조)

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| 구현체에서 `embed()` 오버라이드 | `_do_embed()`를 오버라이드. `embed()`는 부모의 공통 메서드 |
| `_do_embed()`에서 빈 리스트 체크/자동 load | 부모 `embed()`가 처리. `_do_embed()`는 순수 임베딩 로직만 |
| 모듈 최상단에서 `import openai` | `load()` 내에서 lazy import |
| `__init__.py`에 새 임베더 import 누락 | 파일 추가 즉시 `__init__.py` 업데이트 |
| API 키를 코드에 하드코딩 | `os.environ["OPENAI_API_KEY"]`로 `.env`에서 읽기 |
| `_do_embed()` 반환 길이 ≠ 입력 길이 | `len(결과) == len(texts)` 반드시 보장 |
| 구현체에서 `logger` 선언 제거 | 향후 세부 로그 추가 대비로 유지 |
| ada-002에 dimensions 파라미터 전달 | ada-002는 dimensions 미지원, 조건 분기 필요 |

---

## 관련 파일 참조

- 전체 아키텍처: `CLAUDE.md`
- 범용 Registry 구현: `backend/ingestion/base_registry.py`
- 도메인 모델: `backend/models/chunk.py`
- Chunker 가이드: `.claude/backend/ingestion/chunker/SKILL.md`
- Pipeline Orchestrator: `backend/ingestion/pipeline.py`
- 환경변수: `backend/.env`
- config.yaml: `backend/config/config.yaml`
- 테스트: `backend/tests/ingestion/test_embedder.py`
