# Election Radar

2026년 6월 3일 재보궐선거(평택을, 부산북구갑) 판세를 실시간 분석하는 웹 서비스.

뉴스 자동 크롤링 → 텍스트 청킹 → 벡터 임베딩 → VectorDB 저장 → RAG 기반 판세 판정 → 대시보드 시각화

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 백엔드 | Python 3.11+, FastAPI, APScheduler |
| 임베딩 | OpenAI text-embedding-3-small (기본), BAAI/bge-m3, KoSimCSE |
| Vector DB | ChromaDB (로컬 개발, 기본), Qdrant (운영) |
| RAG | 자체 구현 (Retriever → Reranker → Scorer) |
| LLM 판정 | OpenAI GPT-4o (기본), Anthropic Claude (대안) |
| 프론트엔드 | Next.js 16, TypeScript, Recharts |
| 인프라 | Docker Compose (멀티스테이지 빌드) |
| 패키지 관리 | uv + pyproject.toml (백엔드), npm (프론트엔드) |

---

## 디렉토리 구조

```
election_expectation/
├── README.md
├── CLAUDE.md
├── OVERVIEW.md                      # 프로젝트 의의·역할·중요성
├── docker-compose.yml               # 백엔드 + 프론트엔드 통합 실행
│
├── backend/
│   ├── Dockerfile                   # Python 3.11 + uv 기반 이미지
│   ├── .dockerignore
│   ├── pyproject.toml
│   ├── .env                         # 비밀값 (git 제외, OPENAI_API_KEY 등)
│   │
│   ├── config/
│   │   └── config.yaml              # 크롤링 스케줄·선거구·후보·컴포넌트·RAG 설정
│   │
│   ├── app/                         # FastAPI 서버
│   │   ├── main.py                  # 진입점 (CORS 미들웨어 포함)
│   │   ├── core/
│   │   │   ├── dependencies.py      # DI 컨테이너 (config, VectorDB)
│   │   │   ├── pipeline_runner.py   # 백그라운드 파이프라인 실행
│   │   │   └── scheduler.py         # APScheduler (cron 자동 수집 + 판정)
│   │   └── api/v1/
│   │       ├── routes/
│   │       │   ├── admin.py         # 관리자 API
│   │       │   └── scores.py        # 판세 결과 API
│   │       └── schemas/
│   │           ├── admin.py         # 관리자 요청/응답 스키마
│   │           └── score.py         # 판세 결과 요청/응답 스키마
│   │
│   ├── data/                        # 수집 결과 저장
│   │   ├── scraped_urls.jsonl       # 수집된 URL 기록 (중복 방지, 영속 누적)
│   │   ├── articles_YYYY-MM-DD_HHMMSS.jsonl
│   │   ├── chunks_YYYY-MM-DD_HHMMSS.jsonl
│   │   ├── embeddings_YYYY-MM-DD_HHMMSS.jsonl
│   │   └── verdicts/               # 판정 결과 (선거구별 JSONL 누적)
│   │       ├── pyeongtaek_b.jsonl
│   │       └── busan_bukgu_gap.jsonl
│   │
│   ├── ingestion/                   # 수집 파이프라인 (scrape→tag→chunk→embed→store)
│   │   ├── pipeline.py
│   │   ├── tagger.py                # 기사 → 후보/선거구 자동 태깅
│   │   ├── base_registry.py
│   │   ├── scraper/
│   │   │   ├── base.py, naver.py, political.py, url_store.py, run.py
│   │   ├── chunker/
│   │   │   ├── base.py, korean_paragraph.py, sentence.py, token.py, semantic.py, recursive.py
│   │   └── embedder/
│   │       ├── base.py, openai_embedder.py, bge.py, ko_simcse.py
│   │
│   ├── vectordb/                    # Vector DB 추상화 (6종)
│   │   ├── base.py, qdrant_repo.py, chroma_repo.py, milvus_repo.py
│   │   ├── lancedb_repo.py, weaviate_repo.py, pgvector_repo.py
│   │
│   ├── rag/                         # 판정 엔진 (retrieve→rerank→score)
│   │   ├── pipeline.py, retriever.py, reranker.py, scorer.py
│   │   ├── openai_scorer.py, anthropic_scorer.py
│   │   └── verdict_store.py         # 판정 결과 JSONL 영속 저장/조회
│   │
│   ├── models/                      # 도메인 Pydantic 모델
│   │   ├── article.py, chunk.py, score.py
│   │
│   └── tests/
│       ├── app/                     # 31개 (admin 20 + scores 6 + scheduler 5)
│       ├── ingestion/               # 108개 (scraper 33 + tagger 19 + chunker 27 + embedder 16 + pipeline 13)
│       ├── rag/                     # 52개 (retriever 15 + reranker 9 + scorer 17 + verdict_store 11)
│       └── vectordb/               # 38개
│
└── frontend/                        # Next.js 대시보드
    ├── Dockerfile                   # Node 20 멀티스테이지 빌드 (standalone)
    ├── .dockerignore
    ├── next.config.ts               # output: "standalone" (Docker 최적화)
    └── src/
        ├── app/                     # 메인 대시보드 + 관리자 페이지
        ├── components/              # VerdictCard, WinProbChart, DistrictSelector
        └── lib/                     # API 클라이언트, 타입 정의
```

---

## VectorDB 저장 전략

파이프라인은 매번 수집할 때마다 기존 VectorDB에 **누적 업데이트(upsert)** 합니다.

### 왜 매번 새로 만들지 않는가?

| | 1안: 매번 신규 생성 | **2안: 기존 DB 업데이트 (채택)** |
|--|--|--|
| 방식 | 컬렉션 삭제 후 재생성 | 기존 컬렉션에 upsert |
| 장점 | 충돌 없음 | 과거 기사 흐름 보존 |
| 단점 | 과거 기사 소실, 매번 전체 임베딩 비용 | 중복/노이즈 관리 필요 |

선거 판세 분석은 **시간에 따른 여론 흐름**이 중요하므로 과거 데이터를 유지하는 2안을 채택하되, 아래 세 가지 안전장치로 충돌을 방지합니다.

### 안전장치

#### (a) 결정적 ID — 중복 벡터 원천 차단

`ChunkWithEmbedding`의 ID를 `sha256(article_url + chunk_index)`로 생성합니다. 같은 기사의 같은 청크는 항상 동일한 ID를 가지므로, upsert 시 기존 벡터를 **덮어쓰기(갱신)** 합니다.

```python
# article_url + chunk_index → 결정적 해시
id = sha256("https://example.com/news/123::chunk::0")[:16]
```

파이프라인을 여러 번 돌려도 중복이 발생하지 않습니다.

#### (b) 시간 필터 — RAG 검색 시 최근 N일 기사만 사용

Retriever가 검색 결과를 `published_at` 기준으로 후처리 필터링합니다. 오래된 기사("A후보 출마 선언" 등)가 최신 판세 분석에 노이즈로 작용하는 것을 방지합니다.

```yaml
rag:
  retriever:
    lookback_days: 14    # 최근 14일 기사만 사용 (null이면 전체)
```

```bash
# CLI에서 오버라이드 가능
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --lookback-days 7
```

#### (c) 만료 정리 — 오래된 벡터 주기적 삭제

설정된 일수 이전의 벡터를 VectorDB에서 물리적으로 삭제합니다. 수집 파이프라인 완료 후 자동 실행되며, RAG 파이프라인에서도 수동 실행할 수 있습니다.

```yaml
rag:
  purge_days: 60         # 60일 이전 벡터 삭제 (null이면 비활성)
```

```bash
# 수동 삭제
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --purge-days 30
```

### 데이터 흐름 요약

```
수집(scrape) → 태깅(tag) → 청킹(chunk) → 임베딩(embed) → 저장(store)
  ├── URL 중복 방지: scraped_urls.jsonl (영속 누적)
  ├── 자동 태깅: 기사 제목+본문 키워드 매칭 → candidate/district_id 자동 부여
  ├── JSONL: 매번 타임스탬프 파일 신규 생성 (이력 보관)
  └── VectorDB: 기존 컬렉션에 upsert (결정적 ID로 중복 방지)
       ├── 검색 시: lookback_days로 최근 기사만 사용
       └── 정리 시: purge_days 이전 벡터 자동 삭제
```

---

## 빠른 시작

### 방법 A: Docker Compose (권장)

가장 간단한 실행 방법입니다. Docker와 Docker Compose만 설치되어 있으면 됩니다.

```bash
# 1. API 키 설정
echo "OPENAI_API_KEY=sk-proj-..." > backend/.env

# 2. 빌드 및 실행
docker compose up --build
```

- 대시보드: http://localhost:3000
- 관리자 페이지: http://localhost:3000/admin
- API 문서 (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health

```bash
# 백그라운드 실행
docker compose up --build -d

# 로그 확인
docker compose logs -f backend
docker compose logs -f frontend

# 종료
docker compose down
```

수집 데이터(`data/`)와 VectorDB(`.chroma/`)는 Docker 볼륨에 영속 저장됩니다.

### 방법 B: 로컬 직접 실행

#### 1. 환경 세팅

```bash
# 백엔드
cd backend
uv sync

# 프론트엔드
cd frontend
npm install
```

#### 2. API 키 설정

`backend/.env` 파일에 API 키를 설정합니다:

```
OPENAI_API_KEY=sk-proj-...
```

##### 3. 수집 파이프라인 실행

```bash
# 전체 파이프라인 (scrape → tag → chunk → embed → store)
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

#### 4. RAG 판정 파이프라인 실행

```bash
# 평택을 판정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b

# 부산북구갑 판정
PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap

# 검색 수 조정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --top-k 10

# 최근 7일 기사만 사용
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --lookback-days 7

# 검색 결과만 확인 (LLM 판정 생략)
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --skip-score

# 30일 이전 벡터 삭제 후 판정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --purge-days 30
```

#### 5. 스크레이퍼 단독 실행 (디버깅용)

```bash
PYTHONPATH=. python -m ingestion.scraper.run
PYTHONPATH=. python -m ingestion.scraper.run --scraper naver --days 5
```

#### 6. 서버 + 프론트엔드 실행

백엔드와 프론트엔드를 **두 터미널에서 동시에** 실행해야 합니다.
프론트엔드(`localhost:3000`)가 백엔드(`localhost:8000`) API를 호출하므로 백엔드가 먼저 실행되어 있어야 합니다.
백엔드에는 CORS 미들웨어가 설정되어 `localhost:3000`에서의 요청을 허용합니다.

```bash
# 백엔드 (터미널 1)
cd backend
uv run uvicorn app.main:app --reload
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

```bash
# 프론트엔드 (터미널 2)
cd frontend
npm install   # 최초 1회
npm run dev
```

- 대시보드: http://localhost:3000
- 관리자 페이지: http://localhost:3000/admin

관리자 페이지에서 스크레이퍼 선택(전체/네이버/정치 매체) + 기간 입력 후 **"수집 실행"** 버튼으로 파이프라인을 수동 실행할 수 있습니다.
기간을 비워두면 config.yaml의 `lookback_days` 기본값이 적용됩니다.

#### 7. 테스트 실행

```bash
# 백엔드 전체 테스트 (215개 passed, 14개 skipped)
cd backend
PYTHONPATH=. pytest tests/ -v

# 모듈별
PYTHONPATH=. pytest tests/ingestion/test_scraper.py -v
PYTHONPATH=. pytest tests/ingestion/test_chunker.py -v
PYTHONPATH=. pytest tests/ingestion/test_embedder.py -v
PYTHONPATH=. pytest tests/ingestion/test_pipeline.py -v
PYTHONPATH=. pytest tests/vectordb/test_repository.py -v
PYTHONPATH=. pytest tests/rag/ -v
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
| 청크 | `data/chunks_YYYY-MM-DD_HHMMSS.jsonl` | 청킹 결과 |
| 임베딩 | `data/embeddings_YYYY-MM-DD_HHMMSS.jsonl` | 임베딩 결과 |

---

## 설정 (config.yaml)

`backend/config/config.yaml`에서 다음을 설정합니다:

- **schedule**: 크롤링 주기 (cron 표현식)
- **districts**: 선거구별 후보 및 검색 키워드
- **scrapers**: 스크레이퍼별 설정 (max_articles, delay, lookback_days)
- **chunker / embedder / vectordb**: 파이프라인 컴포넌트 설정
- **rag**: 검색(top_k, lookback_days), 재정렬(min_score), 판정(provider, model), 만료 정리(purge_days)

### 후보 명단 (2026년 4월 기준)

**평택을**: 김용남(더불어민주당), 조국(조국혁신당), 유의동(국민의힘), 김재연(진보당), 황교안(자유와혁신)

**부산북구갑**: 하정우(더불어민주당), 한동훈(무소속), 박민식(국민의힘)

---

## 청커 (Chunker)

| 청커 | config.yaml 키 | 외부 라이브러리 | 용도 |
|------|----------------|----------------|------|
| KoreanParagraphChunker | `korean_paragraph` | 없음 | **기본값 권장**, 일반 뉴스 기사 |
| SentenceChunker | `sentence` | `kss` | 문장 경계 중시 |
| TokenChunker | `token` | `tiktoken` | LLM 토큰 제한 정밀 제어 |
| SemanticChunker | `semantic` | `sentence-transformers` | 긴 기사, 주제 전환 감지 |
| RecursiveChunker | `recursive` | 없음 | 구분자 커스터마이징 |

---

## 임베더 (Embedder)

| 임베더 | config.yaml 키 | 외부 라이브러리 | 차원 | 특징 |
|--------|----------------|----------------|------|------|
| OpenAIEmbedder | `openai` | `openai` | 1536 (small) / 3072 (large) | **기본값**, API 기반, 빠름 |
| BGEM3Embedder | `bge_m3` | `FlagEmbedding` | 1024 | 로컬 추론, ~2GB 모델 |
| KoSimCSEEmbedder | `ko_simcse` | `sentence-transformers` | 768 | 한국어 특화, ~500MB 모델 |

---

## VectorDB

| VectorDB | config.yaml 키 | 인프라 | 특징 |
|----------|----------------|--------|------|
| ChromaRepository | `chroma` | 없음 (로컬 파일) | **개발 환경 기본값**, 서버 불필요 |
| QdrantRepository | `qdrant` | Docker | **운영 환경 권장**, 고성능 |
| MilvusLiteRepository | `milvus_lite` | 없음 (SQLite) | pip만으로 사용 가능 |
| LanceDBRepository | `lancedb` | 없음 (파일) | 가장 경량 |
| WeaviateRepository | `weaviate` | Docker | GraphQL, 하이브리드 검색 |
| PgvectorRepository | `pgvector` | PostgreSQL | 기존 PG 인프라 활용 |

---

## RAG 판정 엔진

```
retrieve (VectorDB 검색 + 시간 필터)
  → rerank (임계값 필터링 + URL 중복 제거 + 점수 정렬)
    → score (LLM 판정 + 승리 확률 정규화)
```

| 컴포넌트 | 설정 | 설명 |
|---------|------|------|
| Retriever | `top_k: 20`, `lookback_days: 14` | 후보당 검색 수, 최근 N일 필터, 필터 fallback |
| Reranker | `min_score: 0.3`, `deduplicate: true` | 유사도 임계값, 동일 기사 중복 제거 |
| Scorer | `provider: openai`, `model: gpt-4o` | LLM 판정 (openai / anthropic), `json_object` 모드 |

### Scorer 구현체

| Scorer | config 키 | 모델 | 특징 |
|--------|----------|------|------|
| OpenAIScorer | `openai` | GPT-4o | **기본값**, `json_object` 응답 모드 |
| AnthropicScorer | `anthropic` | Claude | API 키 추가 시 전환 가능 |

### RAG 로깅 정책

| 컴포넌트 | WARNING | INFO | DEBUG |
|---------|---------|------|-------|
| Retriever | 검색 결과 변환 실패 | fallback 재검색, 시간 필터 적용, 검색 완료 건수 | 개별 결과 (id, score, title) |
| Reranker | 빈 입력 | 재정렬 완료 (전후 건수) | 임계값/중복 제거 건수 |
| Scorer | 0건 입력, 파싱 실패, 확률 합 ≠ 1.0 | LLM 요청/응답 (소요 시간), 판정 완료 | 프롬프트/응답 전문 |

---

## 아키텍처 패턴

### Strategy + Registry

모든 교체 가능 컴포넌트(Scraper, Chunker, Embedder, VectorDB, Scorer)는 동일한 패턴을 따릅니다:

```
ComponentRegistry.register("name")  →  config.yaml type 값과 매칭
ComponentRegistry.create("name")    →  인스턴스 생성
```

`config.yaml`의 `type` 또는 `provider` 값만 변경하면 구현체가 전환됩니다.

### Lazy Import

무거운 라이브러리(httpx, chromadb, qdrant_client, kss, FlagEmbedding, anthropic 등)는 구현체 `__init__` 또는 `load()` 내부에서 lazy import합니다. 사용하지 않는 구현체의 패키지가 미설치여도 다른 컴포넌트에 영향 없음.

---

## 네이버 셀렉터 (2026년 4월 기준)

네이버 검색 결과는 SDS 디자인 시스템으로 전환되어 `data-heatmap-target` 속성 기반 셀렉터를 사용합니다.
수집 결과가 0건이면 네이버 HTML 구조가 또 변경되었을 수 있으니, 실제 페이지를 확인하고 `naver.py` 상단 셀렉터 상수를 업데이트하세요.

---

## Docker 구성

### 구조

| 파일 | 역할 |
|------|------|
| `docker-compose.yml` | backend + frontend 통합 실행 |
| `backend/Dockerfile` | Python 3.11-slim + uv, FastAPI 서버 |
| `frontend/Dockerfile` | Node 20-alpine 멀티스테이지 빌드 (deps → build → standalone) |

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | (필수) | `backend/.env`에 설정 |
| `CORS_ORIGINS` | `http://localhost:3000` | 허용할 프론트엔드 origin (콤마 구분) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api/v1` | 프론트엔드가 호출할 백엔드 API 주소 |

### 볼륨

| 볼륨 | 컨테이너 경로 | 용도 |
|------|---------------|------|
| `backend-data` | `/app/data` | 수집 기사, 청크, 임베딩, 판정 결과 JSONL |
| `backend-chroma` | `/app/.chroma` | ChromaDB 영속 저장 |

### 운영 배포 시 변경 사항

서버 도메인에 맞게 환경변수를 조정합니다:

```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - CORS_ORIGINS=https://yourdomain.com
  frontend:
    build:
      args:
        NEXT_PUBLIC_API_URL: https://yourdomain.com/api/v1
```

---

## 관리자 API

서버 실행: `cd backend && uv run uvicorn app.main:app --reload`

- Swagger UI: http://localhost:8000/docs
- CORS: `localhost:3000` 허용 (프론트엔드 연동)

### 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/v1/admin/pipeline/run` | 수집 파이프라인 백그라운드 실행 (scraper, days 지정 가능) |
| `POST` | `/api/v1/admin/pipeline/rebuild` | VectorDB 삭제 → 전체 파이프라인 재실행 |
| `GET` | `/api/v1/admin/pipeline/status` | 현재 파이프라인 실행 상태 (running/completed/failed/idle) |
| `GET` | `/api/v1/admin/vectordb/stats` | VectorDB 타입, 컬렉션명, 저장 건수 |
| `POST` | `/api/v1/admin/vectordb/purge` | N일 이전 벡터 수동 삭제 |
| `GET` | `/api/v1/admin/config` | 전체 config.yaml 설정 조회 |
| `GET` | `/api/v1/admin/config/rag` | RAG 설정만 조회 |
| `PATCH` | `/api/v1/admin/config/rag` | RAG 설정 변경 (lookback_days, top_k, scorer 등) |
| `GET` | `/api/v1/admin/districts` | 선거구/후보 목록 |

### 판세 결과 API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/scores/{district_id}/latest` | 최신 판정 결과 (후보별 verdict, 승률, 근거) |
| `GET` | `/api/v1/scores/{district_id}/history` | 판정 이력 조회 (date_from, date_to, limit) |
| `GET` | `/api/v1/scores/{district_id}/timeseries` | 시계열 차트용 데이터 (날짜별 후보 승률) |
| `POST` | `/api/v1/scores/{district_id}/run` | 판정 실행 (LLM 호출 → 결과 저장) |

### 자동 스케줄

서버 시작 시 config.yaml의 `schedule.cron` 설정에 따라 APScheduler가 자동으로 수집 + 판정을 실행합니다.
기본값: `0 7,12,18 * * *` (하루 3회)

### 사용 예시

```bash
# 파이프라인 실행 (네이버만, 최근 5일)
curl -X POST http://localhost:8000/api/v1/admin/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"scraper": "naver", "days": 5}'

# 실행 상태 확인
curl http://localhost:8000/api/v1/admin/pipeline/status

# VectorDB 통계
curl http://localhost:8000/api/v1/admin/vectordb/stats

# RAG 설정 변경 (lookback_days → 7일, scorer → anthropic)
curl -X PATCH http://localhost:8000/api/v1/admin/config/rag \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 7, "scorer_provider": "anthropic", "scorer_model": "claude-sonnet-4-6"}'

# 만료 벡터 정리 (30일 이전)
curl -X POST http://localhost:8000/api/v1/admin/vectordb/purge \
  -H "Content-Type: application/json" \
  -d '{"purge_days": 30}'
```

---

## 현재 구현 상태

- [x] 프로젝트 구조 설계, Strategy + Registry 패턴
- [x] Scraper 구현 완료 (NaverNewsScraper, PoliticalNewsScraper, URL 영속 저장소)
- [x] Chunker 구현 완료 (5종: korean_paragraph, sentence, token, semantic, recursive)
- [x] Embedder 구현 완료 (3종: openai, bge_m3, ko_simcse)
- [x] 기사 → 후보/선거구 자동 태깅 (키워드 매칭 기반)
- [x] IngestionPipeline 연결 (scrape→tag→chunk→embed→store)
- [x] VectorDB Repository 구현 완료 (6종: qdrant, chroma, milvus_lite, lancedb, weaviate, pgvector)
- [x] VectorDB 안전장치 (결정적 ID, 시간 필터, 만료 정리)
- [x] RAG 판정 엔진 구현 완료 (Retriever, Reranker, Scorer)
- [x] OpenAIScorer (GPT-4o) + AnthropicScorer (Claude)
- [x] 판정 결과 영속 저장 (VerdictStore — 선거구별 JSONL 이력 누적)
- [x] FastAPI 관리자 API (파이프라인 실행, VectorDB 관리, 설정 변경)
- [x] 판세 결과 API (최신/이력/시계열 조회, 판정 실행)
- [x] APScheduler 연동 (cron 주기 자동 수집 + 판정)
- [x] 테스트 215개 passed, 14개 skipped
- [x] TypeScript 대시보드 (Next.js 16 + Recharts)
- [x] CORS 미들웨어 (프론트엔드 → 백엔드 API 연동, `CORS_ORIGINS` 환경변수 지원)
- [x] Docker 배포 (백엔드 Dockerfile + 프론트엔드 멀티스테이지 Dockerfile + docker-compose.yml)
