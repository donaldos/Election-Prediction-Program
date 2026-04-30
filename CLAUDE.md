# Election Radar — CLAUDE.md

이 파일은 Claude Code가 프로젝트 전반을 이해하기 위한 컨텍스트 문서입니다.
새 대화를 시작할 때마다 이 파일을 먼저 읽고 아래 구조와 규칙을 숙지하세요.

---

## 프로젝트 개요

**목적**: 2026년 6월 3일 재보궐선거(평택을, 부산북구갑)의 판세를 실시간 분석하는 일반 대중용 웹 서비스.

**핵심 기능**:
1. 네이버 뉴스 및 정치 뉴스 자동 크롤링 (config.yaml 스케줄 기반)
2. 기사 청킹 → 임베딩 → Vector DB 저장
3. RAG 시스템으로 후보별 우세/균형/열세 판정 + 승리 확률 산출
4. 여론조사 PDF 수동 업로드 및 지지도 수치 추출
5. TypeScript 대시보드에 시계열 판세 차트 + 근거 목록 제공

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 백엔드 | Python 3.11+, FastAPI, APScheduler |
| 임베딩 | OpenAI text-embedding-3-small (기본), BAAI/bge-m3, KoSimCSE (대안) |
| Vector DB | ChromaDB (로컬 개발, 현재 기본), Qdrant (운영) |
| RAG | 자체 구현 (Retriever → Reranker → Scorer) |
| LLM 판정 | OpenAI GPT-4o (기본), Anthropic Claude (대안) |
| PDF 파싱 | pdfplumber |
| 한국어 NLP | kss (문장 분리), kiwipiepy |
| 프론트엔드 | Next.js 14, TypeScript, Recharts |
| 인프라 | Docker Compose (Qdrant + backend) |
| 패키지 관리 | uv + pyproject.toml |

---

## 디렉토리 구조

```
election-radar/
├── CLAUDE.md                        ← 이 파일
├── README.md
├── docker-compose.yml
│
├── backend/
│   ├── pyproject.toml
│   ├── .env                         ← 비밀값 (git 제외, OPENAI_API_KEY 등)
│   │
│   ├── config/
│   │   └── config.yaml              ← 크롤링 스케줄·선거구·후보·컴포넌트 타입·RAG 설정
│   │
│   ├── app/                         ← FastAPI 진입점 (미구현)
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── dependencies.py      ← DI 컨테이너 (컴포넌트 싱글톤 관리)
│   │   │   └── scheduler.py         ← APScheduler (크롤링 주기 실행)
│   │   └── api/v1/
│   │       ├── routes/
│   │       │   ├── articles.py
│   │       │   └── scores.py
│   │       └── schemas/
│   │           ├── article.py
│   │           └── score.py
│   │
│   ├── data/                            ← 수집 결과 저장
│   │   ├── scraped_urls.jsonl           ← URL 기록 (중복 방지, 영속 누적)
│   │   ├── articles_YYYY-MM-DD_HHMMSS.jsonl    ← 수집 기사
│   │   ├── chunks_YYYY-MM-DD_HHMMSS.jsonl      ← 청킹 결과
│   │   └── embeddings_YYYY-MM-DD_HHMMSS.jsonl  ← 임베딩 결과
│   │
│   ├── ingestion/                   ← 수집 파이프라인 (scrape→chunk→embed→store)
│   │   ├── pipeline.py              ← Orchestrator CLI
│   │   ├── base_registry.py         ← 범용 ComponentRegistry[T]
│   │   │
│   │   ├── scraper/                 ← 뉴스 수집
│   │   │   ├── base.py              ← AbstractScraper (ABC) + ScraperRegistry
│   │   │   ├── naver.py             ← NaverNewsScraper (data-heatmap-target 기반)
│   │   │   ├── political.py         ← PoliticalNewsScraper (RSS 파싱)
│   │   │   ├── url_store.py         ← ScrapedUrlStore (URL JSONL 영속 저장)
│   │   │   └── run.py               ← 수동 실행 CLI 스크립트
│   │   │
│   │   ├── chunker/                 ← 텍스트 청킹
│   │   │   ├── base.py              ← AbstractChunker (ABC) + ChunkerRegistry
│   │   │   ├── korean_paragraph.py  ← KoreanParagraphChunker (문단 기반, 기본값)
│   │   │   ├── sentence.py          ← SentenceChunker (kss 문장 분리)
│   │   │   ├── token.py             ← TokenChunker (tiktoken 토큰 기준)
│   │   │   ├── semantic.py          ← SemanticChunker (임베딩 유사도 경계 감지)
│   │   │   └── recursive.py         ← RecursiveChunker (재귀적 구분자 분리)
│   │   │
│   │   └── embedder/                ← 벡터 임베딩
│   │       ├── base.py              ← AbstractEmbedder (ABC) + EmbedderRegistry + embed_query()
│   │       ├── openai_embedder.py   ← OpenAIEmbedder (API 기반, 기본값)
│   │       ├── bge.py               ← BGEM3Embedder (로컬 추론, 1024차원)
│   │       └── ko_simcse.py         ← KoSimCSEEmbedder (한국어 특화, 768차원)
│   │
│   ├── vectordb/                    ← Vector DB 추상화
│   │   ├── base.py                  ← AbstractVectorRepository (ABC) + VectorRepositoryRegistry
│   │   ├── qdrant_repo.py           ← QdrantRepository (Docker, 운영 환경)
│   │   ├── chroma_repo.py           ← ChromaRepository (로컬 내장, 개발 환경, 현재 기본)
│   │   ├── milvus_repo.py           ← MilvusLiteRepository (SQLite 기반)
│   │   ├── lancedb_repo.py          ← LanceDBRepository (파일 기반, 경량)
│   │   ├── weaviate_repo.py         ← WeaviateRepository (Docker, GraphQL)
│   │   └── pgvector_repo.py         ← PgvectorRepository (PostgreSQL 확장)
│   │
│   ├── rag/                         ← 판정 엔진 (retrieve→rerank→score)
│   │   ├── pipeline.py              ← RAG 파이프라인 CLI
│   │   ├── retriever.py             ← Retriever (VectorDB 검색 → SearchResult 변환)
│   │   ├── reranker.py              ← Reranker (임계값 필터링 + 중복 제거 + 정렬)
│   │   ├── scorer.py                ← AbstractScorer (ABC) + ScorerRegistry + 프롬프트 구성
│   │   ├── openai_scorer.py         ← OpenAIScorer (GPT-4o, 기본값)
│   │   └── anthropic_scorer.py      ← AnthropicScorer (Claude)
│   │
│   ├── models/                      ← 도메인 Pydantic 모델 (공유)
│   │   ├── article.py               ← RawArticle, Article
│   │   ├── chunk.py                 ← Chunk, ChunkWithEmbedding
│   │   └── score.py                 ← SearchResult, CandidateScore, DailyVerdict
│   │
│   └── tests/
│       ├── ingestion/
│       │   ├── test_scraper.py      ← 33개
│       │   ├── test_chunker.py      ← 27개 (19 passed, 8 skipped)
│       │   ├── test_embedder.py     ← 16개 (14 passed, 2 skipped)
│       │   └── test_pipeline.py     ← 13개
│       ├── rag/
│       │   ├── test_retriever.py    ← 12개
│       │   ├── test_reranker.py     ← 9개
│       │   └── test_scorer.py       ← 17개
│       └── vectordb/
│           └── test_repository.py   ← 33개 (29 passed, 4 skipped)
│
├── frontend/                        ← (미구현)
│   └── src/
│       ├── app/
│       ├── components/
│       └── lib/
│
└── .claude/                         ← Claude Code 컴포넌트별 스킬 가이드
    └── backend/
        ├── ingestion/
        │   ├── scraper/SKILL.md
        │   ├── chunker/SKILL.md
        │   └── embedder/SKILL.md
        ├── vectordb/SKILL.md
        └── rag/SKILL.md
```

---

## 핵심 아키텍처 패턴

### 1. Strategy + Registry 패턴
모든 교체 가능 컴포넌트(Scraper / Chunker / Embedder / VectorRepository / Scorer)는 동일한 구조를 따릅니다.

```
ComponentRegistry (base_registry.py)
    ├── AbstractXxx (base.py)          ← ABC, 인터페이스 정의
    └── ConcreteXxx (구현파일.py)      ← @Registry.register("name") 데코레이터로 자동 등록
```

**구현체 전환 방법**: `config.yaml`의 `type` 또는 `provider` 값만 변경. 코드 수정 불필요.

```yaml
# 예: VectorDB 전환
vectordb:
  type: chroma          # qdrant → chroma 변경만으로 전환 완료

# 예: LLM 판정 프로바이더 전환
rag:
  scorer:
    provider: anthropic  # openai → anthropic 변경만으로 전환 완료
    model: claude-sonnet-4-6
```

### 2. Lazy Import 규칙
무거운 ML 라이브러리(`FlagEmbedding`, `chromadb`, `qdrant_client`, `kss`, `anthropic` 등)는
반드시 구현체 `__init__` 또는 `load()` 내부에서 lazy import합니다.
사용하지 않는 구현체의 패키지가 설치되지 않아도 다른 컴포넌트는 정상 동작해야 합니다.

```python
# ✅ 올바른 방법
def __init__(self):
    from FlagEmbedding import BGEM3FlagModel
    self._model = BGEM3FlagModel(...)

# ❌ 잘못된 방법 (모듈 최상단 import)
from FlagEmbedding import BGEM3FlagModel
```

### 3. 도메인 모델 불변 규칙
`models/` 하위 Pydantic 모델은 레이어 간 데이터 계약입니다.
필드 변경 시 반드시 **모든 레이어의 영향 범위를 확인**하고 PR에 명시하세요.

```
RawArticle → (scraper 출력)
Article    → (전처리 완료, pipeline 내부 사용)
Chunk      → (chunker 출력)
ChunkWithEmbedding → (embedder 출력, vectordb 입력)
SearchResult       → (retriever 출력, reranker 입출력)
CandidateScore     → (scorer 출력, API 응답)
DailyVerdict       → (scorer 출력, API 응답 단위)
```

---

## 두 파이프라인

### 1. 수집 파이프라인 (ingestion)

```
scrape → chunk → embed → store (VectorDB)
```

```bash
PYTHONPATH=. python -m ingestion.pipeline                       # 전체
PYTHONPATH=. python -m ingestion.pipeline --scraper naver       # 네이버만
PYTHONPATH=. python -m ingestion.pipeline --days 5              # 5일 전부터
PYTHONPATH=. python -m ingestion.pipeline --skip-embed          # 임베딩 생략
PYTHONPATH=. python -m ingestion.pipeline --skip-store          # VectorDB 저장 생략
PYTHONPATH=. python -m ingestion.pipeline --skip-chunk          # 청킹·임베딩·저장 생략
```

### 2. RAG 판정 파이프라인

```
retrieve → rerank → score (LLM 판정)
```

```bash
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b     # 평택을 판정
PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap  # 부산북구갑 판정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --top-k 10
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --skip-score  # 검색만
```

---

## config.yaml 전체 구조

```yaml
schedule:
  cron: "0 7,12,18 * * *"          # 하루 3회 실행

districts:
  - id: pyeongtaek_b
    name: 평택을
    candidates:
      - name: 김용남
        party: 더불어민주당
        keywords: ["김용남", "평택을", "평택 재보궐"]
      # ... (5명)

  - id: busan_bukgu_gap
    name: 부산북구갑
    candidates:
      # ... (3명)

scrapers:
  naver:
    type: naver
    params:
      max_articles_per_run: 100
      request_delay_sec: 1.5
      lookback_days: 2
  political:
    type: political
    params:
      urls: [...]
      max_articles_per_run: 100
      lookback_days: 2

chunker:
  type: korean_paragraph            # korean_paragraph | sentence | token | semantic | recursive
  params:
    chunk_size: 400
    overlap: 50

embedder:
  type: openai                      # openai | bge_m3 | ko_simcse
  params:
    model: text-embedding-3-small
    dimensions: 1536
    batch_size: 100

vectordb:
  type: chroma                      # qdrant | chroma | milvus_lite | lancedb | weaviate | pgvector
  collection: election_chunks
  params:
    persist_dir: .chroma

rag:
  retriever:
    top_k: 20
  reranker:
    min_score: 0.3
    deduplicate: true
  scorer:
    provider: openai                # openai | anthropic
    model: gpt-4o
    temperature: 0.1
    max_tokens: 2000
```

---

## 개발 명령어

```bash
# 환경 세팅 (uv)
cd backend
uv sync

# FastAPI 서버 실행
uv run uvicorn app.main:app --reload

# 수집 파이프라인
PYTHONPATH=. python -m ingestion.pipeline
PYTHONPATH=. python -m ingestion.pipeline --scraper naver --days 5

# RAG 판정 파이프라인
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b
PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap --skip-score

# 스크레이퍼 수동 실행 (디버깅용)
PYTHONPATH=. python -m ingestion.scraper.run --scraper naver --days 5

# 테스트
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. pytest tests/rag/ -v
PYTHONPATH=. pytest tests/ingestion/test_scraper.py -v
PYTHONPATH=. pytest tests/vectordb/test_repository.py -v
```

---

## 컴포넌트별 SKILL.md 위치

새 컴포넌트를 구현하기 전에 해당 SKILL.md를 반드시 읽으세요.

| 컴포넌트 | SKILL.md 경로 |
|----------|---------------|
| Scraper  | `.claude/backend/ingestion/scraper/SKILL.md` |
| Chunker  | `.claude/backend/ingestion/chunker/SKILL.md` |
| Embedder | `.claude/backend/ingestion/embedder/SKILL.md` |
| VectorDB | `.claude/backend/vectordb/SKILL.md` |
| RAG      | `.claude/backend/rag/SKILL.md` |

---

## 코드 품질 규칙

- **타입 힌트 필수**: 모든 함수 시그니처에 타입 힌트 작성
- **Pydantic v2 사용**: `model_dump()`, `model_validate()` 사용 (`dict()` 사용 금지)
- **예외 처리**: 스크레이퍼의 네트워크 오류는 `logger.warning`으로 기록 후 계속 진행. 치명적 오류만 raise
- **로깅**: `logging` 표준 라이브러리 사용, `print()` 사용 금지
- **테스트**: 외부 의존성(HTTP, DB, LLM)은 반드시 mock 처리. 실제 네트워크 호출 테스트 금지
- **환경 변수**: `.env` 파일 사용, `python-dotenv`로 로드 (pipeline CLI에서 `load_dotenv()`)

---

## 알려진 제한 사항

- **기사 메타데이터 미매칭**: 수집된 기사의 `candidate`, `district_id` 필드가 빈 문자열. 기사 본문 키워드 기반 후보/선거구 태깅 로직 미구현. Retriever에서 필터 검색 0건 시 필터 없이 재검색하는 fallback으로 우회 중.
- **네이버 셀렉터 변동**: 네이버 SDS 클래스명이 주기적으로 변경됨. `data-heatmap-target`과 `sds-comps-profile-info-*` 시맨틱 클래스 기반으로 2026-04-30 수정 완료. 수집 결과가 0건이면 HTML 구조 변경 가능성 확인 필요.

---

## 현재 구현 상태

- [x] 아키텍처 설계 완료
- [x] Strategy + Registry 패턴 (`ComponentRegistry`, `base_registry.py`)
- [x] Scraper 구현 완료
  - [x] `NaverNewsScraper` — data-heatmap-target 기반 동적 컨테이너 탐색
  - [x] `PoliticalNewsScraper` — 오마이뉴스·프레시안·미디어오늘 RSS 파싱
  - [x] `ScrapedUrlStore` — URL JSONL 영속 저장 (중복 수집 방지)
  - [x] 테스트 33개 통과
- [x] Chunker 구현 완료
  - [x] 5종 구현 (korean_paragraph, sentence, token, semantic, recursive)
  - [x] Template Method 패턴 — `chunk()` 공통 로깅 + `_do_chunk()` 구현체 분리
  - [x] 테스트 27개 (19 passed, 8 skipped)
- [x] Embedder 구현 완료
  - [x] 3종 구현 (openai, bge_m3, ko_simcse)
  - [x] `embed_query()` 메서드 — 단일 텍스트 벡터 변환 (Retriever용)
  - [x] 테스트 16개 (14 passed, 2 skipped)
- [x] IngestionPipeline 연결 (scrape→chunk→embed→store)
  - [x] CLI: `--scraper`, `--days`, `--skip-chunk`, `--skip-embed`, `--skip-store`
  - [x] `load_dotenv()` 적용
  - [x] 테스트 13개 통과
- [x] VectorRepository 구현 완료
  - [x] 6종 구현 (qdrant, chroma, milvus_lite, lancedb, weaviate, pgvector)
  - [x] ChromaDB 다중 필터 `$and` 래퍼 적용
  - [x] 테스트 33개 (29 passed, 4 skipped)
- [x] RAG 판정 엔진 구현 완료
  - [x] `Retriever` — VectorDB 검색 → SearchResult 변환, 필터 fallback
  - [x] `Reranker` — 유사도 임계값 필터링 + URL 중복 제거 + 점수 정렬
  - [x] `Scorer` — Strategy + Registry 패턴, LLM 판정 + 확률 정규화
  - [x] `OpenAIScorer` — GPT-4o (json_object 모드)
  - [x] `AnthropicScorer` — Claude (API 키 추가 시 전환 가능)
  - [x] RAG 파이프라인 CLI (`rag.pipeline`)
  - [x] `SearchResult`, `CandidateScore`, `DailyVerdict` 도메인 모델
  - [x] 테스트 38개 (retriever 12 + reranker 9 + scorer 17)
- [ ] 기사 → 후보/선거구 자동 태깅 (candidate, district_id 매칭)
- [ ] FastAPI 라우터
- [ ] TypeScript 대시보드
