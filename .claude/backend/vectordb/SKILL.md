---
name: vectordb
description: >
  Election Radar 프로젝트의 VectorDB 컴포넌트를 구현하거나 수정할 때 사용.
  QdrantRepository, ChromaRepository, MilvusLiteRepository, LanceDBRepository,
  WeaviateRepository, PgvectorRepository 추가·변경,
  AbstractVectorRepository 인터페이스 수정, VectorRepositoryRegistry 등록,
  VectorDB 설정(config.yaml) 변경, VectorDB 테스트 작성 시 반드시 이 파일을 먼저 읽으세요.
  Embedder에서 메모리로 전달된 list[ChunkWithEmbedding]을 입력으로 받습니다.
---

# VectorDB 컴포넌트 가이드

## 역할과 데이터 흐름

Embedder가 생성한 `list[ChunkWithEmbedding]`을 메모리에서 전달받아
벡터 저장소에 저장하고, 이후 RAG 검색 시 유사 벡터를 반환합니다.

```
Embedder.embed() → list[ChunkWithEmbedding]
        ↓
AbstractVectorRepository.upsert(list[ChunkWithEmbedding]) → 저장
AbstractVectorRepository.search(query_vector)              → RAG 검색
        ↓
RAG Retriever → Scorer
```

**VectorDB는 벡터 저장/검색만** 담당합니다.
벡터 생성은 Embedder, 판정 로직은 RAG가 처리합니다.

---

## 파일 구조

```
vectordb/
├── base.py              ← AbstractVectorRepository ABC + VectorRepositoryRegistry (반드시 먼저 읽기)
├── qdrant_repo.py       ← QdrantRepository (Docker, 운영 환경)
├── chroma_repo.py       ← ChromaRepository (로컬 내장, 개발 환경)
├── milvus_repo.py       ← MilvusLiteRepository (SQLite 기반)
├── lancedb_repo.py      ← LanceDBRepository (파일 기반, 경량)
├── weaviate_repo.py     ← WeaviateRepository (Docker, GraphQL)
├── pgvector_repo.py     ← PgvectorRepository (PostgreSQL 확장)
└── __init__.py          ← 6개 구현체 import → Registry 자동 등록
```

> **`__init__.py` 필수**: 새 구현체 추가 시 반드시 import 추가.
> import가 없으면 `@VectorRepositoryRegistry.register` 데코레이터가 실행되지 않아 등록 누락.

```python
# vectordb/__init__.py
from vectordb import qdrant_repo   # QdrantRepository 등록
from vectordb import chroma_repo   # ChromaRepository 등록
from vectordb import milvus_repo   # MilvusLiteRepository 등록
from vectordb import lancedb_repo  # LanceDBRepository 등록
from vectordb import weaviate_repo # WeaviateRepository 등록
from vectordb import pgvector_repo # PgvectorRepository 등록
```

---

## 도메인 모델

VectorDB가 입력으로 받는 `ChunkWithEmbedding`의 구조:

```python
# models/chunk.py (이미 구현됨)

class ChunkWithEmbedding(Chunk):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float]

    @property
    def metadata(self) -> dict:
        return self.model_dump(exclude={"id", "embedding"})
```

- `id`: 벡터 고유 식별자 (UUID)
- `embedding`: 임베딩 벡터 (list[float])
- `metadata`: id, embedding 제외 전체 필드 (VectorDB payload로 저장)

---

## AbstractVectorRepository 인터페이스 (Template Method 패턴)

공개 메서드 `upsert()`, `search()`, `delete()`, `count()`는 부모의 **concrete 메서드**로
공통 로깅과 자동 load()를 처리합니다.
구현체는 `_do_upsert()`, `_do_search()`, `_do_delete()`, `_do_count()`만 오버라이드합니다.

```python
# vectordb/base.py

import logging
from abc import ABC, abstractmethod
from ingestion.base_registry import ComponentRegistry
from models.chunk import ChunkWithEmbedding

logger = logging.getLogger(__name__)


class AbstractVectorRepository(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    def upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        if not chunks:
            logger.warning("[%s] 빈 벡터 리스트 — 스킵", self.name)
            return 0
        if not self.is_loaded:
            self.load()
        logger.info("[%s] upsert 시작 — %d개 벡터", self.name, len(chunks))
        count = self._do_upsert(chunks)
        logger.info("[%s] upsert 완료 — %d개 저장", self.name, count)
        return count

    @abstractmethod
    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int: ...

    def search(self, query_vector: list[float], top_k: int = 10, filters: dict | None = None) -> list[dict]:
        if not self.is_loaded:
            self.load()
        logger.info("[%s] 검색 시작 — top_k=%d", self.name, top_k)
        results = self._do_search(query_vector, top_k, filters)
        logger.info("[%s] 검색 완료 — %d개 결과", self.name, len(results))
        return results

    @abstractmethod
    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]: ...

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        if not self.is_loaded:
            self.load()
        logger.info("[%s] 삭제 시작 — %d개", self.name, len(ids))
        count = self._do_delete(ids)
        logger.info("[%s] 삭제 완료 — %d개", self.name, count)
        return count

    @abstractmethod
    def _do_delete(self, ids: list[str]) -> int: ...

    def count(self) -> int:
        if not self.is_loaded:
            self.load()
        return self._do_count()

    @abstractmethod
    def _do_count(self) -> int: ...


VectorRepositoryRegistry = ComponentRegistry(AbstractVectorRepository, "VectorRepository")
```

### 호출 흐름

```
외부 호출 → upsert(list[ChunkWithEmbedding])    ← AbstractVectorRepository (공통)
               │
               ├── 빈 리스트 체크 → WARNING 로그 + return 0
               ├── is_loaded 체크 → 미로드 시 자동 load()
               ├── INFO 로그: upsert 시작
               │
               └── _do_upsert(chunks)              ← 구현체 (오버라이드)
                      │
                      └── int (저장 건수) 반환
               │
               └── INFO 로그: upsert 완료
```

---

## 구현체별 상세 명세

### 1. QdrantRepository `qdrant` (운영 기본값)

**인프라**: Docker (`docker compose up -d qdrant`)
**Lazy Load 대상**: `qdrant_client`

```python
@VectorRepositoryRegistry.register("qdrant")
class QdrantRepository(AbstractVectorRepository):

    def __init__(self, collection="election_chunks", host="localhost", port=6333, dimensions=1536):
        ...

    def load(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        # 컬렉션 미존재 시 자동 생성 (COSINE 거리)

    def _do_upsert(self, chunks):
        from qdrant_client.models import PointStruct
        # PointStruct(id, vector, payload) 변환 후 upsert

    def _do_search(self, query_vector, top_k, filters):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        # filters dict → FieldCondition 변환
```

**config.yaml:**
```yaml
vectordb:
  type: qdrant
  collection: election_chunks
  params:
    host: localhost
    port: 6333
    dimensions: 1536
```

---

### 2. ChromaRepository `chroma` (개발 환경 권장)

**인프라**: 없음 (로컬 파일, pip install만으로 사용)
**Lazy Load 대상**: `chromadb`

```python
@VectorRepositoryRegistry.register("chroma")
class ChromaRepository(AbstractVectorRepository):

    def __init__(self, collection="election_chunks", persist_dir=".chroma"):
        ...

    def load(self):
        import chromadb
        # PersistentClient + get_or_create_collection (cosine)

    def _do_upsert(self, chunks):
        # collection.upsert(ids, embeddings, metadatas, documents)
        # published_at → str 변환 필요 (Chroma는 datetime 미지원)
```

**config.yaml:**
```yaml
vectordb:
  type: chroma
  collection: election_chunks
  params:
    persist_dir: .chroma
```

---

### 3. MilvusLiteRepository `milvus_lite`

**인프라**: 없음 (SQLite 기반, pip install만으로 사용)
**Lazy Load 대상**: `pymilvus`

```python
@VectorRepositoryRegistry.register("milvus_lite")
class MilvusLiteRepository(AbstractVectorRepository):

    def __init__(self, collection="election_chunks", db_path="./milvus_lite.db", dimensions=1536):
        ...

    def load(self):
        from pymilvus import MilvusClient
        # 컬렉션 스키마 정의 (VARCHAR 필드 + FLOAT_VECTOR)
        # COSINE 인덱스 자동 생성
```

**config.yaml:**
```yaml
vectordb:
  type: milvus_lite
  collection: election_chunks
  params:
    db_path: ./milvus_lite.db
    dimensions: 1536
```

---

### 4. LanceDBRepository `lancedb`

**인프라**: 없음 (파일 기반, 가장 경량)
**Lazy Load 대상**: `lancedb`, `pyarrow`

```python
@VectorRepositoryRegistry.register("lancedb")
class LanceDBRepository(AbstractVectorRepository):

    def __init__(self, collection="election_chunks", db_path="./lancedb_data", dimensions=1536):
        ...

    def load(self):
        import lancedb
        # connect → 기존 테이블 open 또는 첫 upsert 시 create

    def _do_upsert(self, chunks):
        # 테이블 미존재 시 첫 데이터로 create_table
        # 이후 add()로 추가
```

**config.yaml:**
```yaml
vectordb:
  type: lancedb
  collection: election_chunks
  params:
    db_path: ./lancedb_data
    dimensions: 1536
```

---

### 5. WeaviateRepository `weaviate`

**인프라**: Docker (weaviate 서버)
**Lazy Load 대상**: `weaviate`

```python
@VectorRepositoryRegistry.register("weaviate")
class WeaviateRepository(AbstractVectorRepository):

    def __init__(self, collection="ElectionChunks", host="localhost", port=8080, grpc_port=50051, dimensions=1536):
        ...

    def load(self):
        import weaviate
        # connect_to_local → 컬렉션 미존재 시 생성 (Vectorizer.none)

    def _do_upsert(self, chunks):
        # batch.dynamic() 사용, deterministic UUID로 중복 방지
```

**config.yaml:**
```yaml
vectordb:
  type: weaviate
  collection: ElectionChunks
  params:
    host: localhost
    port: 8080
    grpc_port: 50051
    dimensions: 1536
```

---

### 6. PgvectorRepository `pgvector`

**인프라**: PostgreSQL + pgvector 확장
**Lazy Load 대상**: `psycopg`, `pgvector`

```python
@VectorRepositoryRegistry.register("pgvector")
class PgvectorRepository(AbstractVectorRepository):

    def __init__(self, collection="election_chunks", dsn="postgresql://localhost:5432/election_radar", dimensions=1536):
        ...

    def load(self):
        import psycopg
        from pgvector.psycopg import register_vector
        # CREATE EXTENSION vector + CREATE TABLE + IVFFlat 인덱스

    def _do_search(self, query_vector, top_k, filters):
        # cosine 거리: embedding <=> query_vector
        # 1 - distance = similarity score
```

**config.yaml:**
```yaml
vectordb:
  type: pgvector
  collection: election_chunks
  params:
    dsn: "postgresql://localhost:5432/election_radar"
    dimensions: 1536
```

---

## load() 설계 원칙

```
pipeline.py 시작
    │
    ├── repo.load()          ← 연결·컬렉션 초기화 (1회만)
    │      │
    │      ├── QdrantRepository:     Qdrant 서버 연결, 컬렉션 생성
    │      ├── ChromaRepository:     PersistentClient 생성, 컬렉션 생성
    │      ├── MilvusLiteRepository: SQLite DB 연결, 스키마·인덱스 생성
    │      ├── LanceDBRepository:    파일 DB 연결, 테이블 open
    │      ├── WeaviateRepository:   Weaviate 서버 연결, 컬렉션 생성
    │      └── PgvectorRepository:   PostgreSQL 연결, 테이블·인덱스 생성
    │
    └── repo.upsert(embedded_chunks)   ← 이미 로드됨, 즉시 실행
```

**규칙**:
- 공개 메서드 내부에서 `is_loaded` 체크 후 미로드 시 자동 `load()` 호출
- `load()`는 **멱등(idempotent)**: 여러 번 호출해도 중복 연결하지 않음
- 컬렉션/테이블 미존재 시 `load()` 또는 첫 `_do_upsert()` 에서 자동 생성
- 구현체는 `_do_*()` 메서드만 오버라이드. 공개 메서드를 오버라이드하면 로깅 누락

---

## 구현체별 비교표

| 구현체 | Registry 키 | 외부 패키지 | 인프라 | 비용 | 최적 상황 |
|--------|------------|------------|--------|------|----------|
| QdrantRepository | `qdrant` | `qdrant-client` | Docker | 무료 (셀프호스팅) | **운영 기본값**, 고성능, 필터링 강력 |
| ChromaRepository | `chroma` | `chromadb` | 없음 | 무료 | **개발/테스트**, 서버 불필요 |
| MilvusLiteRepository | `milvus_lite` | `pymilvus` | 없음 | 무료 | SQLite 기반, pip만으로 사용 |
| LanceDBRepository | `lancedb` | `lancedb` | 없음 | 무료 | 가장 경량, 빠른 프로토타이핑 |
| WeaviateRepository | `weaviate` | `weaviate-client` | Docker | 무료 (셀프호스팅) | GraphQL, 하이브리드 검색 |
| PgvectorRepository | `pgvector` | `psycopg`, `pgvector` | PostgreSQL | 무료 | 기존 PG 인프라 재활용 |

---

## search() 반환 형식

모든 구현체의 `search()`는 통일된 dict 리스트를 반환합니다:

```python
[
    {
        "id": "chunk-uuid",
        "score": 0.95,           # 코사인 유사도 (0~1, 높을수록 유사)
        "text": "평택을 선거구...",
        "article_url": "https://...",
        "source": "naver_news",
        "title": "판세 분석",
        "candidate": "후보A",
        "district_id": "pyeongtaek_b",
        ...
    },
]
```

**filters 파라미터**: dict 형태로 필드명=값 필터링 지원.

```python
# 특정 선거구만 검색
results = repo.search(
    query_vector=embedding,
    top_k=10,
    filters={"district_id": "pyeongtaek_b"},
)
```

---

## pipeline.py 연동 방식

```python
# ingestion/pipeline.py (store 관련 부분)

import vectordb  # noqa: F401
from vectordb.base import VectorRepositoryRegistry

class IngestionPipeline:
    def _store(self, embedded: list[ChunkWithEmbedding]) -> int:
        cfg = self._config.get("vectordb", {})
        repo = VectorRepositoryRegistry.create(
            cfg.get("type", "chroma"),
            collection=cfg.get("collection", "election_chunks"),
            **cfg.get("params", {}),
        )
        repo.load()
        count = repo.upsert(embedded)
        return count
```

---

## 로깅 규칙

### 공통 로깅 (AbstractVectorRepository — 자동 처리)

| 레벨 | 내용 |
|------|------|
| `WARNING` | 빈 벡터 리스트 입력 시 스킵 |
| `INFO` | upsert/search/delete 시작 및 완료 (건수) |

### 구현체별 로깅

| 레벨 | 내용 |
|------|------|
| `INFO` | load() 완료 (연결 정보, 컬렉션명), 컬렉션 생성 시 알림 |

> **구현체별 세부 로그 추가 대비**: 모든 구현체에 `logger = logging.getLogger(__name__)` 선언 유지.

---

## 테스트 작성 가이드

외부 DB 연결이 필요한 테스트는 mock 또는 skip 처리합니다.

```python
# tests/vectordb/test_repository.py

from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from models.chunk import ChunkWithEmbedding

SAMPLE_CHUNKS = [
    ChunkWithEmbedding(
        id="chunk-001",
        text="평택을 선거구 여론조사 결과",
        chunk_index=0, char_count=14,
        article_url="https://example.com/1",
        source="naver_news", title="판세 분석",
        published_at=datetime(2026, 5, 1),
        candidate="후보A", district_id="pyeongtaek_b",
        chunker_type="korean_paragraph",
        embedding=[0.1] * 1536,
    ),
]


# ── Registry 테스트 ──────────────────────────────────────

def test_all_registered():
    import vectordb
    from vectordb.base import VectorRepositoryRegistry
    names = VectorRepositoryRegistry.registered_names
    assert "qdrant" in names
    assert "chroma" in names
    assert "milvus_lite" in names
    assert "lancedb" in names
    assert "weaviate" in names
    assert "pgvector" in names


def test_create_unknown_raises():
    from vectordb.base import VectorRepositoryRegistry
    with pytest.raises(ValueError, match="미등록"):
        VectorRepositoryRegistry.create("nonexistent")


# ── Qdrant mock 테스트 ───────────────────────────────────

def test_qdrant_upsert():
    from vectordb.qdrant_repo import QdrantRepository
    repo = QdrantRepository()
    repo._loaded = True
    repo._client = MagicMock()

    with patch.dict("sys.modules", {
        "qdrant_client": MagicMock(),
        "qdrant_client.models": MagicMock(),
    }):
        count = repo.upsert(SAMPLE_CHUNKS)
        assert count == 1


# ── Chroma mock 테스트 ───────────────────────────────────

def test_chroma_search():
    from vectordb.chroma_repo import ChromaRepository
    repo = ChromaRepository()
    repo._loaded = True
    repo._collection = MagicMock()
    repo._collection.query.return_value = {
        "ids": [["id1"]], "distances": [[0.1]],
        "metadatas": [[{"candidate": "후보A"}]],
        "documents": [["test"]],
    }
    results = repo.search([0.1] * 1536, top_k=5)
    assert len(results) == 1


# ── 외부 DB 필요 (skip) ─────────────────────────────────

@pytest.mark.skip(reason="pymilvus 설치 필요")
def test_milvus_lite_load(): ...

@pytest.mark.skip(reason="lancedb 설치 필요")
def test_lancedb_load(): ...

@pytest.mark.skip(reason="weaviate 서버 필요")
def test_weaviate_load(): ...

@pytest.mark.skip(reason="PostgreSQL + pgvector 필요")
def test_pgvector_load(): ...
```

### mock 패턴 — 외부 패키지 미설치 환경

qdrant_client, chromadb 등 미설치 환경에서는 `patch.dict("sys.modules", ...)` 으로
모듈을 mock 합니다.

```python
# qdrant_client 미설치 환경에서 upsert 테스트
with patch.dict("sys.modules", {
    "qdrant_client": MagicMock(),
    "qdrant_client.models": MagicMock(PointStruct=MagicMock()),
}):
    count = repo.upsert(SAMPLE_CHUNKS)
```

---

## 새 VectorDB 구현 체크리스트

### 1단계: 구현 파일 생성

```python
# vectordb/my_repo.py

@VectorRepositoryRegistry.register("my_db")
class MyDBRepository(AbstractVectorRepository):

    def __init__(self, collection: str, **kwargs) -> None:
        ...

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        # 저장 로직만. 로깅·빈 입력 체크는 부모 upsert()가 처리.
        ...

    def _do_search(self, query_vector, top_k, filters) -> list[dict]:
        # 반환 형식: [{"id": ..., "score": ..., "text": ..., ...}]
        ...

    def _do_delete(self, ids: list[str]) -> int: ...
    def _do_count(self) -> int: ...
```

### 2단계: `__init__.py`에 등록

```python
from vectordb import my_repo  # ← 추가
```

### 3단계: config.yaml 추가

```yaml
vectordb:
  type: my_db
  collection: election_chunks
  params:
    host: localhost
    port: 9999
```

### 4단계: 테스트 작성 (`tests/vectordb/test_repository.py` 참조)

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| 구현체에서 `upsert()` 오버라이드 | `_do_upsert()`를 오버라이드. `upsert()`는 부모의 공통 메서드 |
| `_do_upsert()`에서 빈 리스트 체크/자동 load | 부모 `upsert()`가 처리 |
| 모듈 최상단에서 `import qdrant_client` | `load()` 또는 `_do_*()` 내에서 lazy import |
| `__init__.py`에 새 구현체 import 누락 | 파일 추가 즉시 `__init__.py` 업데이트 |
| `search()` 반환 형식 불일치 | `[{"id": ..., "score": ..., ...}]` dict 리스트 통일 |
| datetime을 그대로 payload에 저장 | `str(published_at)` 변환 (Chroma, LanceDB 등) |
| 컬렉션 미존재 시 에러 | `load()` 에서 자동 생성 (get_or_create 패턴) |
| 구현체에서 `logger` 선언 제거 | 향후 세부 로그 추가 대비로 유지 |

---

## 관련 파일 참조

- 전체 아키텍처: `CLAUDE.md`
- 범용 Registry 구현: `backend/ingestion/base_registry.py`
- 도메인 모델: `backend/models/chunk.py`
- Embedder 가이드: `.claude/backend/ingestion/embedder/SKILL.md`
- Pipeline Orchestrator: `backend/ingestion/pipeline.py`
- config.yaml: `backend/config/config.yaml`
- 테스트: `backend/tests/vectordb/test_repository.py`
