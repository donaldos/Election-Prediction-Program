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
│   ├── app/                         ← FastAPI 진입점
│   │   ├── main.py                  ← FastAPI app + lifespan (APScheduler 자동 시작)
│   │   ├── core/
│   │   │   ├── dependencies.py      ← DI 컨테이너 (config 캐시, VectorDB 싱글톤)
│   │   │   ├── pipeline_runner.py   ← 백그라운드 파이프라인 실행 관리 (스레드 기반)
│   │   │   └── scheduler.py         ← APScheduler (cron 주기 자동 수집 + 판정)
│   │   └── api/v1/
│   │       ├── routes/
│   │       │   ├── admin.py         ← 관리자 API (파이프라인 실행, VectorDB 관리, 설정 변경)
│   │       │   └── scores.py        ← 판세 결과 API (최신/이력/시계열 조회, 판정 실행)
│   │       └── schemas/
│   │           ├── admin.py         ← 관리자 요청/응답 스키마
│   │           └── score.py         ← 판세 결과 요청/응답 스키마
│   │
│   ├── data/                            ← 수집 결과 저장
│   │   ├── scraped_urls.jsonl           ← URL 기록 (중복 방지, 영속 누적)
│   │   ├── articles_YYYY-MM-DD_HHMMSS.jsonl    ← 수집 기사
│   │   ├── chunks_YYYY-MM-DD_HHMMSS.jsonl      ← 청킹 결과
│   │   ├── embeddings_YYYY-MM-DD_HHMMSS.jsonl  ← 임베딩 결과
│   │   └── verdicts/                    ← 판정 결과 (선거구별 JSONL 누적)
│   │       ├── pyeongtaek_b.jsonl
│   │       └── busan_bukgu_gap.jsonl
│   │
│   ├── ingestion/                   ← 수집 파이프라인 (scrape→tag→chunk→embed→store)
│   │   ├── pipeline.py              ← Orchestrator CLI
│   │   ├── tagger.py                ← 기사 → 후보/선거구 자동 태깅 (키워드 매칭 + 문맥 검증)
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
│   │   │   ├── recursive.py         ← RecursiveChunker (재귀적 구분자 분리)
│   │   │   └── article_aware.py     ← ArticleAwareChunker (기사 구조 인식, 제목 prefix + 리드 분리)
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
│   │   ├── pgvector_repo.py         ← PgvectorRepository (PostgreSQL 확장)
│   │   └── pinecone_repo.py        ← PineconeRepository (Pinecone Cloud, SaaS)
│   │
│   ├── rag/                         ← 판정 엔진 (retrieve→rerank→score)
│   │   ├── pipeline.py              ← RAG 파이프라인 CLI
│   │   ├── retriever.py             ← Retriever (VectorDB 검색 → SearchResult 변환)
│   │   ├── reranker.py              ← Reranker (임계값 필터링 + 중복 제거 + 정렬)
│   │   ├── scorer.py                ← AbstractScorer (ABC) + ScorerRegistry + 프롬프트 구성
│   │   ├── openai_scorer.py         ← OpenAIScorer (GPT-4o, 기본값)
│   │   ├── anthropic_scorer.py      ← AnthropicScorer (Claude)
│   │   ├── verdict_graph.py         ← LangGraph 판정 오케스트레이션 (검증·일관성·보정)
│   │   └── verdict_store.py         ← VerdictStore (판정 결과 JSONL 영속 저장/조회)
│   │
│   ├── models/                      ← 도메인 Pydantic 모델 (공유)
│   │   ├── article.py               ← RawArticle, Article
│   │   ├── chunk.py                 ← Chunk, ChunkWithEmbedding
│   │   └── score.py                 ← SearchResult, CandidateScore, DailyVerdict
│   │
│   └── tests/
│       ├── ingestion/
│       │   ├── test_scraper.py      ← 40개
│       │   ├── test_chunker.py      ← 37개 (29 passed, 8 skipped)
│       │   ├── test_embedder.py     ← 16개 (14 passed, 2 skipped)
│       │   └── test_pipeline.py     ← 20개
│       ├── rag/
│       │   ├── test_retriever.py    ← 12개
│       │   ├── test_reranker.py     ← 9개
│       │   ├── test_scorer.py       ← 17개
│       │   └── test_verdict_store.py ← 11개
│       └── vectordb/
│           └── test_repository.py   ← 48개 (43 passed, 5 skipped)
│
├── frontend/                        ← Next.js 대시보드
│   └── src/
│       ├── app/
│       │   ├── page.tsx             ← 메인 대시보드 (선거구 선택 + 판세 + 차트)
│       │   └── admin/page.tsx       ← 관리자 페이지 (파이프라인, VectorDB, RAG 설정)
│       ├── components/
│       │   ├── VerdictCard.tsx       ← 후보별 판정 결과 카드
│       │   ├── WinProbChart.tsx      ← 승률 시계열 차트 (Recharts)
│       │   └── DistrictSelector.tsx  ← 선거구 탭 선택
│       └── lib/
│           ├── api.ts               ← API 클라이언트 (fetch 래퍼)
│           └── types.ts             ← TypeScript 타입 정의
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

## 설계 결정: LangChain 미사용, 자체 모듈 구성

본 프로젝트는 LangChain 프레임워크를 사용하지 않고, 각 파이프라인 단계를 자체 모듈로 구현한다.

### 이유

**1. 컴포넌트 자유 교체 (Strategy + Registry)**

LangChain은 자체 추상화(`Document`, `BaseRetriever`, `VectorStore` 등)에 구현체를 맞춰야 한다.
본 프로젝트는 Scraper 2종, Chunker 6종, Embedder 3종, VectorDB 7종, Scorer 2종을
`config.yaml`의 `type` 값 하나로 전환할 수 있는 Strategy + Registry 패턴을 직접 구현했다.
LangChain의 래퍼 계층을 거치지 않으므로, ChromaDB의 `$and` 다중 필터나
`published_at_ts` 숫자 범위 필터 같은 DB별 고유 기능을 제약 없이 활용할 수 있다.

**2. 한국어 처리 최적화**

- 청킹: `kss` 문장 분리, 문단 경계 인식, 기사 구조(제목·리드·본문) 인식 등 한국어 특화 전략을 직접 구현
- 태깅: 키워드 주변 ±50자 문맥 검증으로 동음이의어 오태깅("조국" 등) 방지
- 임베딩: `KoSimCSE`, `BGE-M3` 등 한국어 특화 모델을 LangChain 래퍼 없이 직접 통합

LangChain의 `RecursiveCharacterTextSplitter` 등은 영어 중심 설계로,
한국어 문장 경계·조사·어미 처리에 부적합한 경우가 많다.

**3. 의존성 최소화 + Lazy Import**

LangChain은 수십 개의 하위 패키지를 연쇄 설치한다.
본 프로젝트는 실제 사용하는 라이브러리만 설치하고, 무거운 ML 라이브러리는 Lazy Import로 처리하여
미사용 구현체의 패키지가 없어도 전체 시스템이 정상 동작한다.

**4. LangGraph만 선택적 사용**

유일하게 LangGraph(`langgraph>=1.2.0`)만 도입하여 판정 오케스트레이션에 활용한다.
RAG 파이프라인의 retrieve → rerank → score 각 단계는 자체 구현하되,
**다단계 검증·재시도가 필요한 판정 워크플로우**만 LangGraph `StateGraph`로 관리한다.
`--no-graph` 옵션으로 LangGraph 없이도 동작하므로, LangGraph는 품질 보정 레이어로서 선택적이다.

### 자체 구현 vs LangChain 비교

| 관점 | 자체 구현 (현재) | LangChain 사용 시 |
|------|-----------------|-------------------|
| 컴포넌트 교체 | config.yaml `type` 변경만으로 전환 | 래퍼 계층 경유, DB별 고유 기능 제한적 |
| 한국어 처리 | kss·kiwipiepy·KoSimCSE 직접 통합 | 영어 중심 설계, 커스텀 어려움 |
| 의존성 | 필요한 것만 설치 + Lazy Import | 거대한 의존성 트리 |
| DB 최적화 | 필터·ID 전략 등 DB별 네이티브 기능 직접 활용 | 공통 인터페이스로 추상화, 세부 기능 손실 |
| 유지보수 | 구현체 추가 시 직접 작성 필요 | 커뮤니티 통합 활용 가능 |
| 프로토타이핑 | 초기 구현 비용 높음 | 빠른 PoC 가능 |

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
scrape → tag → chunk → embed → store (VectorDB)
```

- **tag**: 기사 제목+본문에서 config.yaml 후보 키워드를 매칭하여 `candidate`, `district_id` 자동 태깅

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
retrieve (+ 시간 필터) → rerank → score (LLM 판정)
```

```bash
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b     # 평택을 판정
PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap  # 부산북구갑 판정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --top-k 10
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --lookback-days 7  # 최근 7일만
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --purge-days 30    # 30일 이전 벡터 삭제
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
  type: chroma                      # qdrant | chroma | milvus_lite | lancedb | weaviate | pgvector | pinecone
  collection: election_chunks
  params:
    persist_dir: .chroma

rag:
  retriever:
    top_k: 20
    lookback_days: 14                # 최근 N일 기사만 검색 (null이면 전체)
  reranker:
    min_score: 0.3
    deduplicate: true
  scorer:
    provider: openai                # openai | anthropic
    model: gpt-4o
    temperature: 0.1
    max_tokens: 8000
  purge_days: 60                    # N일 이전 벡터 자동 삭제 (null이면 비활성)
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

- **기사 메타데이터 태깅**: `ingestion/tagger.py`로 키워드 매칭 + 문맥 검증 기반 자동 태깅 구현 완료. 키워드 주변 ±50자에 선거 관련 단어가 있어야 유효 매칭으로 인정하여 동음이의어 오태깅("조국" 등) 방지. 키워드에 없는 별명/약칭 사용 시 미태깅될 수 있으므로 config.yaml의 `keywords` 목록을 확장하여 대응.
- **네이버 셀렉터 변동**: 네이버 SDS 클래스명이 주기적으로 변경됨. `data-heatmap-target`과 `sds-comps-profile-info-*` 시맨틱 클래스 기반으로 2026-04-30 수정 완료. 수집 결과가 0건이면 HTML 구조 변경 가능성 확인 필요.

---

## 현재 구현 상태

- [x] 아키텍처 설계 완료
- [x] Strategy + Registry 패턴 (`ComponentRegistry`, `base_registry.py`)
- [x] Scraper 구현 완료
  - [x] `NaverNewsScraper` — data-heatmap-target 기반 동적 컨테이너 탐색
  - [x] `PoliticalNewsScraper` — 오마이뉴스·프레시안·미디어오늘 RSS 파싱
  - [x] `ScrapedUrlStore` — URL JSONL 영속 저장 (중복 수집 방지)
  - [x] `fetch_article_body()` — 기사 상세 페이지 전문 추출 (11개 CSS 셀렉터 + og:description fallback)
  - [x] `_enrich_bodies()` — 수집 후 기사별 상세 페이지 방문하여 전문 교체
  - [x] 테스트 40개 통과
- [x] Chunker 구현 완료
  - [x] 6종 구현 (korean_paragraph, sentence, token, semantic, recursive, article_aware)
  - [x] Template Method 패턴 — `chunk()` 공통 로깅 + `_do_chunk()` 구현체 분리
  - [x] ArticleAwareChunker — 기사 구조 인식 (제목 prefix + 리드 분리 + 문단 경계 분할)
  - [x] 테스트 37개 (29 passed, 8 skipped)
- [x] Embedder 구현 완료
  - [x] 3종 구현 (openai, bge_m3, ko_simcse)
  - [x] `embed_query()` 메서드 — 단일 텍스트 벡터 변환 (Retriever용)
  - [x] 테스트 16개 (14 passed, 2 skipped)
- [x] 기사 → 후보/선거구 자동 태깅 구현 완료
  - [x] `ingestion/tagger.py` — 키워드 매칭 + 문맥 검증 기반 candidate/district_id 자동 태깅
  - [x] 문맥 검증: 키워드 주변 ±50자에 선거 관련 단어 존재 여부 확인 (동음이의어 오태깅 방지)
  - [x] 단일 후보 매칭 → candidate + district_id, 다수 후보 → district_id만, 비관련 → 미태깅
  - [x] 수집 파이프라인에 scrape → **tag** → chunk 단계로 통합
  - [x] 테스트 28개 통과
- [x] IngestionPipeline 연결 (scrape→tag→chunk→embed→store)
  - [x] CLI: `--scraper`, `--days`, `--skip-chunk`, `--skip-embed`, `--skip-store`
  - [x] `load_dotenv()` 적용
  - [x] `_filter_chunks()` — 50자 미만 및 미태깅 청크 필터링
  - [x] 테스트 20개 통과
- [x] VectorRepository 구현 완료
  - [x] 7종 구현 (qdrant, chroma, milvus_lite, lancedb, weaviate, pgvector, pinecone)
  - [x] ChromaDB 다중 필터 `$and` 래퍼 + `published_at_ts` 숫자 필터
  - [x] 결정적 ID (`sha256(article_url + chunk_index)`) — 중복 벡터 원천 차단
  - [x] 만료 정리 (`delete_older_than()`) — 오래된 벡터 물리 삭제
  - [x] 테스트 48개 (43 passed, 5 skipped)
- [x] RAG 판정 엔진 구현 완료
  - [x] `Retriever` — VectorDB 검색 → SearchResult 변환, 필터 fallback, `lookback_days` 시간 필터
  - [x] `Reranker` — 유사도 임계값 필터링 + URL 중복 제거 + 점수 정렬
  - [x] `Scorer` — Strategy + Registry 패턴, LLM 판정 + 확률 정규화 + 응답 시간 로깅
  - [x] `OpenAIScorer` — GPT-4o (json_object 모드)
  - [x] `AnthropicScorer` — Claude (API 키 추가 시 전환 가능)
  - [x] RAG 파이프라인 CLI (`rag.pipeline`) — `--lookback-days`, `--purge-days` 옵션
  - [x] `SearchResult`, `CandidateScore`, `DailyVerdict` 도메인 모델
  - [x] 로깅 정책 — WARNING/INFO/DEBUG 3단계 (Retriever, Reranker, Scorer 각각)
  - [x] 테스트 52개 (retriever 15 + reranker 9 + scorer 17 + verdict_store 11)
- [x] LangGraph 판정 오케스트레이션 구현 완료
  - [x] `verdict_graph.py` — score → validate → correct 그래프 워크플로우
  - [x] 3중 검증: 근거 일치, 일관성 (승률 30%p 급변 감지), 확률 범위
  - [x] 검증 실패 시 자동 재판정 (최대 2회), `--no-graph`로 비활성화 가능
  - [x] 노드별 상태 로깅 (진입·완료·분기 결정·오류 상세)
  - [x] 테스트 17개 통과
- [x] 판정 결과 영속 저장 구현 완료
  - [x] `VerdictStore` — 선거구별 JSONL 파일로 판정 이력 누적 저장
  - [x] `save()`, `load_all()`, `load_latest()`, `load_range()`, `list_districts()`, `count()`
  - [x] RAG 파이프라인 CLI에서 판정 후 자동 저장
- [x] FastAPI 관리자 API 구현 완료
  - [x] `POST /api/v1/admin/pipeline/run` — 수집 파이프라인 백그라운드 실행
  - [x] `POST /api/v1/admin/pipeline/rebuild` — VectorDB 전체 재구축
  - [x] `GET /api/v1/admin/pipeline/status` — 파이프라인 실행 상태 조회
  - [x] `GET /api/v1/admin/vectordb/stats` — VectorDB 통계 (저장 건수)
  - [x] `POST /api/v1/admin/vectordb/purge` — 만료 벡터 수동 정리
  - [x] `GET /api/v1/admin/config` — 전체 설정 조회
  - [x] `GET/PATCH /api/v1/admin/config/rag` — RAG 설정 조회/변경
  - [x] `GET /api/v1/admin/districts` — 선거구/후보 목록 조회
  - [x] DI 컨테이너 (`dependencies.py`) — config 캐시, VectorDB 싱글톤
  - [x] `PipelineRunner` — 스레드 기반 백그라운드 실행, 중복 실행 방지
  - [x] 테스트 20개 통과
- [x] 판세 결과 API 구현 완료
  - [x] `GET /api/v1/scores/{district_id}/latest` — 최신 판정 결과
  - [x] `GET /api/v1/scores/{district_id}/history` — 판정 이력 (날짜 필터, limit)
  - [x] `GET /api/v1/scores/{district_id}/timeseries` — 시계열 차트 데이터
  - [x] `POST /api/v1/scores/{district_id}/run` — 판정 실행 (관리자)
  - [x] 테스트 6개 통과
- [x] APScheduler 연동
  - [x] `scheduler.py` — config.yaml `schedule.cron`에 따라 수집 + 판정 자동 실행
  - [x] FastAPI lifespan으로 서버 시작/종료 시 스케줄러 자동 관리
  - [x] 테스트 5개 통과
- [x] TypeScript 대시보드 구현 완료
  - [x] Next.js 16 + TypeScript + Tailwind CSS + Recharts
  - [x] 메인 대시보드 — 선거구 선택, 최신 판정 결과, 승률 시계열 차트
  - [x] 관리자 페이지 — 파이프라인 실행, VectorDB 관리, RAG 설정 변경
  - [x] API 클라이언트 — fetch 기반 타입 안전 API 래퍼
  - [x] 빌드 성공
