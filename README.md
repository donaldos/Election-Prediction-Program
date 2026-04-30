# Election Radar

2026년 6월 3일 재보궐선거(평택을, 부산북구갑) 판세를 실시간 분석하는 웹 서비스.

뉴스 자동 크롤링 → 텍스트 청킹 → 벡터 임베딩 → RAG 기반 판세 판정 → 대시보드 시각화

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 백엔드 | Python 3.11+, FastAPI, APScheduler |
| 임베딩 | OpenAI text-embedding-3-small (기본), BAAI/bge-m3, KoSimCSE |
| Vector DB | Qdrant (운영), ChromaDB (로컬) |
| RAG | LangChain |
| 프론트엔드 | Next.js 14, TypeScript, Recharts |
| 인프라 | Docker Compose |

---

## 디렉토리 구조

```
election_expectation/
├── README.md
├── CLAUDE.md
├── docker-compose.yml
│
├── backend/
│   ├── pyproject.toml
│   ├── requirements.txt
│   │
│   ├── config/
│   │   └── config.yaml             # 크롤링 스케줄·선거구·후보·컴포넌트 설정
│   │
│   ├── data/                        # 수집 결과 저장
│   │   ├── scraped_urls.jsonl       # 수집된 URL 기록 (중복 방지)
│   │   ├── articles_YYYY-MM-DD_HHMMSS.jsonl  # 수집 기사
│   │   ├── chunks_YYYY-MM-DD_HHMMSS.jsonl    # 청킹 결과
│   │   └── embeddings_YYYY-MM-DD_HHMMSS.jsonl # 임베딩 결과
│   │
│   ├── ingestion/
│   │   ├── pipeline.py              # 파이프라인 오케스트레이터 (scrape→chunk→embed)
│   │   ├── base_registry.py         # Strategy + Registry 패턴 (범용)
│   │   ├── scraper/
│   │   │   ├── base.py              # AbstractScraper ABC
│   │   │   ├── naver.py             # NaverNewsScraper (검색 HTML 파싱)
│   │   │   ├── political.py         # PoliticalNewsScraper (RSS 파싱)
│   │   │   ├── url_store.py         # 수집 URL 영속 저장소
│   │   │   └── run.py               # 수동 실행 스크립트
│   │   ├── chunker/
│   │   │   ├── base.py              # AbstractChunker ABC + ChunkerRegistry
│   │   │   ├── korean_paragraph.py  # KoreanParagraphChunker (문단 기반, 기본값)
│   │   │   ├── sentence.py          # SentenceChunker (kss 문장 분리)
│   │   │   ├── token.py             # TokenChunker (tiktoken 토큰 기준)
│   │   │   ├── semantic.py          # SemanticChunker (임베딩 유사도 경계 감지)
│   │   │   └── recursive.py         # RecursiveChunker (재귀적 구분자 분리)
│   │   └── embedder/
│   │       ├── base.py              # AbstractEmbedder ABC + EmbedderRegistry
│   │       ├── openai_embedder.py   # OpenAIEmbedder (API 기반, 기본값)
│   │       ├── bge.py               # BGEM3Embedder (로컬 추론, 1024차원)
│   │       └── ko_simcse.py         # KoSimCSEEmbedder (한국어 특화, 768차원)
│   │
│   ├── vectordb/
│   │   ├── base.py                  # AbstractVectorRepository + VectorRepositoryRegistry
│   │   ├── qdrant_repo.py           # QdrantRepository (Docker, 운영)
│   │   ├── chroma_repo.py           # ChromaRepository (로컬 내장, 개발)
│   │   ├── milvus_repo.py           # MilvusLiteRepository (SQLite 기반)
│   │   ├── lancedb_repo.py          # LanceDBRepository (파일 기반)
│   │   ├── weaviate_repo.py         # WeaviateRepository (Docker, GraphQL)
│   │   └── pgvector_repo.py         # PgvectorRepository (PostgreSQL 확장)
│   │
│   ├── models/
│   │   ├── article.py               # RawArticle, Article
│   │   └── chunk.py                 # Chunk, ChunkWithEmbedding
│   │
│   └── tests/
│       └── ingestion/
│           ├── test_scraper.py      # 33개 테스트 케이스
│           ├── test_chunker.py      # 27개 테스트 케이스
│           ├── test_embedder.py     # 16개 테스트 케이스
│           └── test_pipeline.py     # 13개 테스트 케이스
│       └── vectordb/
│           └── test_repository.py   # 33개 테스트 케이스
│
└── frontend/                        # (예정)
```

---

## 빠른 시작

### 1. 환경 세팅

```bash
cd backend
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. 파이프라인 실행

```bash
# 전체 파이프라인 (scrape → chunk → embed)
PYTHONPATH=. python -m ingestion.pipeline

# 네이버만 수집
PYTHONPATH=. python -m ingestion.pipeline --scraper naver

# 검색 기간 변경 (예: 5일 전부터)
PYTHONPATH=. python -m ingestion.pipeline --days 5

# 임베딩·저장 생략 (스크레이핑 + 청킹만)
PYTHONPATH=. python -m ingestion.pipeline --skip-embed

# VectorDB 저장 생략 (JSONL까지만)
PYTHONPATH=. python -m ingestion.pipeline --skip-store

# 청킹·임베딩·저장 생략 (스크레이핑만)
PYTHONPATH=. python -m ingestion.pipeline --skip-chunk
```

### 3. 스크레이퍼 단독 실행 (디버깅용)

```bash
PYTHONPATH=. python -m ingestion.scraper.run
PYTHONPATH=. python -m ingestion.scraper.run --scraper naver
PYTHONPATH=. python -m ingestion.scraper.run --days 5
```

### 4. 테스트 실행

```bash
# 전체 테스트
PYTHONPATH=. pytest tests/ -v

# 스크레이퍼 테스트만
PYTHONPATH=. pytest tests/ingestion/test_scraper.py -v

# 청커 테스트만
PYTHONPATH=. pytest tests/ingestion/test_chunker.py -v

# 임베더 테스트만
PYTHONPATH=. pytest tests/ingestion/test_embedder.py -v

# 파이프라인 테스트만
PYTHONPATH=. pytest tests/ingestion/test_pipeline.py -v

# VectorDB 테스트만
PYTHONPATH=. pytest tests/vectordb/test_repository.py -v
```

---

## 수집 대상 매체

| 스크레이퍼 | 매체 | 방식 |
|-----------|------|------|
| NaverNewsScraper | 네이버 뉴스 (search.naver.com) | SDS 셀렉터 기반 HTML 파싱 |
| PoliticalNewsScraper | 오마이뉴스 | RSS |
| PoliticalNewsScraper | 프레시안 | RSS |
| PoliticalNewsScraper | 미디어오늘 | RSS / HTML |

---

## 수집 결과 저장

| 파일 | 경로 | 용도 |
|------|------|------|
| URL 기록 | `data/scraped_urls.jsonl` | 중복 수집 방지 (영속) |
| 기사 본문 | `data/articles_YYYY-MM-DD_HHMMSS.jsonl` | 일자별 수집 기사 전문 |

**URL 기록 형식 (JSONL):**
```json
{"url": "https://...", "source": "naver_news", "title": "기사 제목", "scraped_at": "2026-04-28T15:30:00"}
```

**기사 본문 형식 (JSONL):**
```json
{"url": "https://...", "source": "naver_news", "title": "기사 제목", "body": "본문...", "published_at": "2026-04-28T09:00:00", "matched_keywords": ["평택을"]}
```

---

## 설정 (config.yaml)

`backend/config/config.yaml`에서 다음을 설정합니다:

- **schedule**: 크롤링 주기 (cron 표현식)
- **districts**: 선거구별 후보 및 검색 키워드
- **scrapers**: 스크레이퍼별 설정 (max_articles, delay, lookback_days)
- **chunker / embedder / vectordb**: 파이프라인 컴포넌트 설정

### 후보 명단 (2026년 4월 기준)

**평택을**: 김용남(더불어민주당), 조국(조국혁신당), 유의동(국민의힘), 김재연(진보당), 황교안(자유와혁신)

**부산북구갑**: 하정우(더불어민주당), 한동훈(무소속), 박민식(국민의힘)

```yaml
scrapers:
  naver:
    type: naver
    params:
      max_articles_per_run: 100
      request_delay_sec: 1.5
      lookback_days: 2          # 오늘 기준 N일 전부터 검색
```

---

## 네이버 셀렉터 (2026년 4월 기준)

네이버 검색 결과는 SDS 디자인 시스템으로 전환되어 `data-heatmap-target` 속성 기반 셀렉터를 사용합니다.
수집 결과가 0건이면 네이버 HTML 구조가 또 변경되었을 수 있으니, 실제 페이지를 확인하고 `naver.py` 상단 셀렉터 상수를 업데이트하세요.

```python
SEL_ARTICLE_CONTAINER = 'div[class*="qhLRRX"]'          # 기사 컨테이너
SEL_TITLE             = 'a[data-heatmap-target=".tit"]'  # 제목 + URL
SEL_SUMMARY           = 'a[data-heatmap-target=".body"]' # 본문 요약
SEL_PRESS             = 'a[data-heatmap-target=".prof"] span'  # 언론사
SEL_DATE              = 'span.sds-comps-text-ellipsis-1' # 날짜
```

날짜 형식: `"2026.05.01."`, `"5분 전"`, `"2시간 전"`, `"1일 전"` 모두 파싱 지원.

---

## 청커 (Chunker)

Scraper가 수집한 기사 본문을 Chunk 단위로 분할합니다. `config.yaml`의 `chunker.type` 값으로 전환합니다.

| 청커 | config.yaml 키 | 외부 라이브러리 | 용도 |
|------|----------------|----------------|------|
| KoreanParagraphChunker | `korean_paragraph` | 없음 | **기본값 권장**, 일반 뉴스 기사 |
| SentenceChunker | `sentence` | `kss` | 문장 경계 중시 |
| TokenChunker | `token` | `tiktoken` | LLM 토큰 제한 정밀 제어 |
| SemanticChunker | `semantic` | `sentence-transformers` | 긴 기사, 주제 전환 감지 |
| RecursiveChunker | `recursive` | 없음 | 구분자 커스터마이징 |

```yaml
chunker:
  type: korean_paragraph       # 5개 중 선택
  params:
    chunk_size: 400
    overlap: 50
```

### 청커 로깅

`AbstractChunker.chunk()`에서 공통 로깅을 자동 처리합니다.

| 레벨 | 내용 |
|------|------|
| `WARNING` | 빈 텍스트 입력 시 스킵 |
| `INFO` | 청킹 시작 (입력 글자수, 기사 제목), 청킹 완료 (청크 수, 평균/최소/최대 글자수) |
| `DEBUG` | 개별 청크별 인덱스, 글자수, 텍스트 미리보기 |

---

## 임베더 (Embedder)

Chunker가 분할한 Chunk를 벡터로 변환합니다. `config.yaml`의 `embedder.type` 값으로 전환합니다.

| 임베더 | config.yaml 키 | 외부 라이브러리 | 차원 | 특징 |
|--------|----------------|----------------|------|------|
| OpenAIEmbedder | `openai` | `openai` | 1536 (small) / 3072 (large) | **기본값**, API 기반, 빠름 |
| BGEM3Embedder | `bge_m3` | `FlagEmbedding` | 1024 | 로컬 추론, ~2GB 모델 |
| KoSimCSEEmbedder | `ko_simcse` | `sentence-transformers` | 768 | 한국어 특화, ~500MB 모델 |

```yaml
embedder:
  type: openai                      # openai | bge_m3 | ko_simcse
  params:
    model: text-embedding-3-small   # text-embedding-3-small | text-embedding-3-large | text-embedding-ada-002
    dimensions: 1536
    batch_size: 100
```

### API 키 설정

OpenAI 임베더 사용 시 `backend/.env` 파일에 API 키를 설정합니다:

```
OPENAI_API_KEY=sk-proj-...
```

### 임베더 로깅

`AbstractEmbedder.embed()`에서 공통 로깅을 자동 처리합니다.

| 레벨 | 내용 |
|------|------|
| `WARNING` | 빈 청크 리스트 입력 시 스킵 |
| `INFO` | 임베딩 시작 (청크 수), 임베딩 완료 (벡터 수, 차원) |
| `INFO` | 모델 로드 완료 (구현체 `load()`) |
| `DEBUG` | 배치별 API 호출/로컬 추론 진행 상황 (구현체 `_do_embed()`) |

---

## VectorDB

임베딩된 벡터를 저장·검색합니다. `config.yaml`의 `vectordb.type` 값으로 전환합니다.

| VectorDB | config.yaml 키 | 외부 라이브러리 | 인프라 | 특징 |
|----------|----------------|----------------|--------|------|
| QdrantRepository | `qdrant` | `qdrant-client` | Docker | **운영 기본값**, 고성능 |
| ChromaRepository | `chroma` | `chromadb` | 없음 (로컬 파일) | **개발 환경 권장**, 서버 불필요 |
| MilvusLiteRepository | `milvus_lite` | `pymilvus` | 없음 (SQLite) | pip만으로 사용 가능 |
| LanceDBRepository | `lancedb` | `lancedb` | 없음 (파일) | 가장 경량, 서버 불필요 |
| WeaviateRepository | `weaviate` | `weaviate-client` | Docker | GraphQL, 하이브리드 검색 |
| PgvectorRepository | `pgvector` | `psycopg`, `pgvector` | PostgreSQL | 기존 PG 인프라 활용 |

```yaml
vectordb:
  type: qdrant                      # qdrant | chroma | milvus_lite | lancedb | weaviate | pgvector
  collection: election_chunks
  params:
    host: localhost
    port: 6333
    dimensions: 1536
```

---

## 아키텍처 패턴

### Strategy + Registry

모든 교체 가능 컴포넌트는 동일한 패턴을 따릅니다:

```
ComponentRegistry.register("name")  →  config.yaml type 값과 매칭
ComponentRegistry.create("name")    →  인스턴스 생성
```

`config.yaml`의 `type` 값만 변경하면 구현체가 전환됩니다.

### Lazy Import

무거운 라이브러리(httpx, bs4, feedparser, kss, tiktoken, sentence-transformers, FlagEmbedding, openai 등)는 구현체 `load()` 메서드 내부에서 lazy import합니다. 사용하지 않는 구현체의 패키지가 미설치여도 다른 컴포넌트에 영향 없음.

---

## 현재 구현 상태

- [x] 프로젝트 구조 설계
- [x] Strategy + Registry 패턴 (`base_registry.py`)
- [x] NaverNewsScraper 구현
- [x] PoliticalNewsScraper 구현
- [x] URL 영속 저장소 (`url_store.py`)
- [x] 수동 실행 스크립트 (`run.py`)
- [x] 테스트 코드 (33개 통과)
- [x] lookback_days — date_from/date_to 생략 시 오늘 기준 N일 전 자동 설정
- [x] 로깅 (INFO/WARNING/DEBUG 단계별)
- [x] Chunker 구현 완료
  - [x] KoreanParagraphChunker — 문단 기반 분할 + 오버랩 (기본값)
  - [x] SentenceChunker — kss 문장 분리 기반
  - [x] TokenChunker — tiktoken 토큰 기준 분할
  - [x] SemanticChunker — 임베딩 유사도 경계 감지
  - [x] RecursiveChunker — 재귀적 구분자 분리
  - [x] Chunk, ChunkWithEmbedding 도메인 모델
  - [x] 테스트 27개 (19개 통과, 8개 skip — 외부 라이브러리 미설치)
- [x] Embedder 구현 완료
  - [x] OpenAIEmbedder — API 기반 (기본값)
  - [x] BGEM3Embedder — 로컬 추론 (1024차원)
  - [x] KoSimCSEEmbedder — 한국어 특화 (768차원)
  - [x] Template Method 패턴 (공통 로깅)
  - [x] 테스트 16개 (14개 통과, 2개 skip — 로컬 모델 다운로드 필요)
- [ ] VectorDB Repository 구현
- [x] IngestionPipeline 연결 (scrape→chunk→embed→store)
  - [x] CLI 옵션: --scraper, --days, --skip-chunk, --skip-embed, --skip-store
  - [x] 단계별 JSONL 저장 (articles, chunks, embeddings)
  - [x] 테스트 13개 통과
- [x] VectorDB Repository 구현 완료
  - [x] Qdrant, ChromaDB, Milvus Lite, LanceDB, Weaviate, pgvector
  - [x] config.yaml type 전환으로 교체 가능
  - [x] 테스트 33개 (29개 통과, 4개 skip — 외부 서버 필요)
- [ ] RAG 스코어링 엔진
- [ ] FastAPI 라우터
- [ ] TypeScript 대시보드
