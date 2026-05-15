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
│   │   ├── polls.jsonl              # 여론조사 데이터 (관리자 페이지에서 입력)
│   │   ├── articles_YYYY-MM-DD_HHMMSS.jsonl
│   │   ├── chunks_YYYY-MM-DD_HHMMSS.jsonl
│   │   ├── embeddings_YYYY-MM-DD_HHMMSS.jsonl
│   │   └── verdicts/               # 판정 결과 (선거구별 JSONL 누적)
│   │       ├── pyeongtaek_b.jsonl
│   │       └── busan_bukgu_gap.jsonl
│   │
│   ├── ingestion/                   # 수집 파이프라인 (scrape→tag→chunk→embed→store)
│   │   ├── pipeline.py
│   │   ├── tagger.py                # 기사 → 후보/선거구 자동 태깅 (문맥 검증 포함)
│   │   ├── base_registry.py
│   │   ├── scraper/
│   │   │   ├── base.py, naver.py, naver_election.py, political.py, url_store.py, run.py
│   │   ├── chunker/
│   │   │   ├── base.py, korean_paragraph.py, sentence.py, token.py, semantic.py, recursive.py, article_aware.py
│   │   └── embedder/
│   │       ├── base.py, openai_embedder.py, bge.py, ko_simcse.py
│   │
│   ├── vectordb/                    # Vector DB 추상화 (7종)
│   │   ├── base.py, qdrant_repo.py, chroma_repo.py, milvus_repo.py
│   │   ├── lancedb_repo.py, weaviate_repo.py, pgvector_repo.py
│   │   └── pinecone_repo.py
│   │
│   ├── rag/                         # 판정 엔진 (retrieve→rerank→score)
│   │   ├── pipeline.py, retriever.py, reranker.py, scorer.py
│   │   ├── openai_scorer.py, anthropic_scorer.py
│   │   ├── verdict_store.py         # 판정 결과 JSONL 영속 저장/조회
│   │   └── poll_store.py            # 여론조사 데이터 JSONL 영속 저장/조회
│   │
│   ├── models/                      # 도메인 Pydantic 모델
│   │   ├── article.py, chunk.py, score.py, poll.py
│   │
│   └── tests/
│       ├── app/                     # 31개 (admin 20 + scores 6 + scheduler 5)
│       ├── ingestion/               # 127개 (scraper 33 + tagger 28 + chunker 37 + embedder 16 + pipeline 13)
│       ├── rag/                     # 57개 (retriever 15 + reranker 14 + scorer 17 + verdict_store 11)
│       └── vectordb/               # 48개 (43 passed, 5 skipped)
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

## 데이터 품질 최적화

### 기사 구조 인식 청킹 (ArticleAwareChunker)

일반 청커는 기사 구조를 무시하고 텍스트를 고정 크기로 분할하므로, 중간 청크에서 "이 기사가 어떤 후보/선거구에 관한 것인지" 문맥이 유실됩니다. `ArticleAwareChunker`는 기사의 **제목·리드·본문** 구조를 인식하여 청킹합니다.

**동작 방식**:
1. 첫 번째 청크: `[제목] {title}\n[리드] {lead 문단}` — 기사의 핵심 정보를 압축
2. 이후 청크: `[제목] {title}\n{본문 세그먼트}` — 모든 청크에 제목을 prefix로 붙여 문맥 유지
3. 본문은 문단(`\n\n`) 또는 줄바꿈(`\n`) 경계에서 분할하여 의미 단위를 보존

```yaml
chunker:
  type: article_aware          # 기사 구조 인식 청킹 사용
  params:
    chunk_size: 400
    overlap: 50
```

**효과**: VectorDB 검색 시 모든 청크에 제목이 포함되어 있어 "김용남 후보의 공약" 같은 쿼리에 대해 본문 중간 청크도 높은 유사도를 반환합니다.

### 태거 문맥 검증 (동음이의어 오태깅 방지)

"조국"이라는 키워드는 정치인 이름이자 "motherland"를 뜻하는 일반 명사이기도 합니다. 키워드 매칭만으로는 "가슴속에 늘 조국을 품고 살아온 핏줄"같은 기사가 조국 후보 관련으로 오태깅될 수 있습니다.

**해결 방법**: 키워드 매칭 시 주변 ±50자 윈도우 내에 **선거 맥락 단어**가 존재하는지 검증합니다.

```
선거 맥락 단어: 후보, 대표, 의원, 출마, 선거, 재보궐, 지지율, 공약, 판세, ...
               국민의힘, 더불어민주당, 조국혁신당, 진보당, 무소속, ...
               국회, 정치, 보수, 진보, 야당, 여당, 표심, 민심
```

| 텍스트 | 매칭 결과 |
|--------|----------|
| "조국 **후보**가 출마를 선언했다" | 태깅 (±50자 내 "후보" 존재) |
| "가슴속에 늘 조국을 품고 살아온 핏줄" | **미태깅** (±50자 내 선거 맥락 단어 없음) |
| "**조국혁신당**에서 활동하는 조국 대표" | 태깅 (±50자 내 "조국혁신당" 존재) |

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
PINECONE_API_KEY=pcsk_...          # Pinecone 사용 시에만 필요
```

##### 3. 수집 파이프라인 실행

```bash
# 전체 파이프라인 (scrape → tag → chunk → embed → store)
PYTHONPATH=. python -m ingestion.pipeline

# 네이버만 수집
PYTHONPATH=. python -m ingestion.pipeline --scraper naver

# 네이버 선거 페이지만 수집
PYTHONPATH=. python -m ingestion.pipeline --scraper naver_election

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

# 단발 질의 모드 (자유 질문 → VectorDB 검색 → LLM 답변)
PYTHONPATH=. python -m rag.pipeline --query "조국의 평택을 지지율 변화는?"
PYTHONPATH=. python -m rag.pipeline --query "한동훈과 박민식 비교" --top-k 10
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

관리자 페이지에서 **여론조사 데이터**도 입력할 수 있습니다:
1. 선거구 선택 → 조사기관·조사일 입력 → 후보별 지지율(%) 입력 → **"적용"** 클릭
2. 저장된 여론조사는 다음 판정 실행 시 LLM 프롬프트에 자동 포함됩니다.
3. 이력 테이블에서 과거 여론조사 데이터를 확인·삭제할 수 있습니다.

#### 7. 테스트 실행

```bash
# 백엔드 전체 테스트 (262개 passed, 15개 skipped)
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
| NaverElectionScraper | 네이버 선거 (news.naver.com/election/region2026) | 선거 전용 페이지 HTML 파싱 |
| PoliticalNewsScraper | 오마이뉴스 | RSS |
| PoliticalNewsScraper | 프레시안 | RSS |
| PoliticalNewsScraper | 미디어오늘 | RSS / HTML |

---

## 수집 결과 저장

| 파일 | 경로 | 용도 |
|------|------|------|
| URL 기록 | `data/scraped_urls.jsonl` | 중복 수집 방지 (영속) |
| 여론조사 | `data/polls.jsonl` | 조사기관·날짜·후보별 지지율 이력 (관리자 페이지에서 입력) |
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
- **rag**: 검색(top_k, lookback_days), 재정렬(min_score, cross_encoder), 판정(provider, model), 만료 정리(purge_days)

여론조사 데이터는 config.yaml이 아닌 관리자 페이지(또는 API)를 통해 `data/polls.jsonl`에 저장됩니다.

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
| ArticleAwareChunker | `article_aware` | 없음 | 기사 구조 인식 (제목 prefix + 리드 분리) |

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
| PineconeRepository | `pinecone` | Pinecone Cloud (SaaS) | 완전 관리형, API 키 필요 |

---

## RAG 판정 엔진

### 전체 흐름

```
retrieve (VectorDB 의미 검색 + 시간 필터 + 후보별·항목별 그룹핑)
  → rerank (그룹별 임계값 필터링 + URL 중복 제거 + 점수 정렬)
    → score (트리 구조 프롬프트 + 여론조사 ±3%p 오차범위 → LLM 판정 + 승리 확률 정규화)
      → save (VerdictStore — JSONL 영속 저장)
```

### Step 1: Retrieve — 후보별·분석 항목별 그룹 검색

Retriever가 `config/query_templates.json`에 정의된 **선거구별·후보별 맞춤 쿼리**로 VectorDB에서 cosine similarity 기반 의미 검색을 수행하고, 결과를 **후보 → 분석 항목**별로 그룹핑합니다.

**쿼리 구조** (`query_templates.json`):
- `_common`: 선거구 공통 쿼리 (판세, 여론조사 지지율)
- 후보별: 지지율, 공약 반응, 강점, 약점, 이슈, 지지율 추이, 출마 여론, 선거 전략 — 8가지 축

예시 (부산북구갑):

| 구분 | 쿼리 예시 | 필터 | 카테고리 감지 |
|------|----------|------|-------------|
| 공통 | `"부산북구갑 재보궐 판세"` | `district_id=busan_bukgu_gap` | → 판세 |
| 공통 | `"부산북구갑 여론조사 지지율"` | `district_id=busan_bukgu_gap` | → 여론조사 |
| 하정우 | `"하정우 부산북구갑 지지율"`, `"하정우 강점"` 등 8건 | `candidate=하정우` | → 지지율, 강점, ... |
| 한동훈 | `"한동훈 부산북구갑 지지율"`, `"한동훈 강점"` 등 8건 | `candidate=한동훈` | → 지지율, 강점, ... |
| 박민식 | `"박민식 부산북구갑 지지율"`, `"박민식 강점"` 등 8건 | `candidate=박민식` | → 지지율, 강점, ... |

각 쿼리는 임베딩 벡터로 변환 후 VectorDB에 전송되며, 쿼리 내 키워드(`지지율 추이`, `공약`, `강점` 등)로 **분석 카테고리가 자동 감지**되어 그룹핑됩니다. 검색 결과가 0건이면 필터 없이 재검색합니다.

```
그룹핑 결과 구조:
{
  "_common": {"판세": [청크...], "여론조사": [청크...]},
  "하정우": {"지지율": [청크...], "공약 반응": [청크...], "강점": [청크...], ...},
  "한동훈": {"지지율": [청크...], "공약 반응": [청크...], "강점": [청크...], ...},
  "박민식": {"지지율": [청크...], ...}
}
```

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `top_k` | 20 | 쿼리당 최대 검색 수 |
| `lookback_days` | 14 | 최근 N일 기사만 사용 (null이면 전체) |

### Step 2: Rerank — 2단계 재정렬 (bi-encoder 필터 + Cross-encoder 정밀 평가)

Reranker가 **각 그룹(후보 × 분석 항목) 단위로** 2단계 재정렬을 수행합니다. 같은 청크가 여러 카테고리에 관련될 수 있으므로 교차 카테고리 중복 제거는 하지 않습니다.

```
Stage 1: bi-encoder score 기반 필터링
  → min_score(0.3) 미만 제거 → URL 중복 제거 → score 정렬

Stage 2: Cross-encoder 정밀 재정렬 (선택적)
  → 1단계 통과 청크에 대해 (query, chunk_text) 쌍으로 관련성 재평가
  → sigmoid 정규화 후 상위 top_n건만 유지
```

**bi-encoder vs Cross-encoder**:
- bi-encoder: 질의와 청크를 독립적으로 벡터화 → cosine similarity (빠르지만 부정·조건·맥락 구분에 약함)
- Cross-encoder: 질의+청크를 하나의 입력으로 결합 → 관련성 직접 평가 (느리지만 정확)

Cross-encoder는 그룹별로 쿼리를 재구성하여 적용합니다:
- `_common` 그룹: `"{선거구명} {카테고리}"` (예: "평택을 판세")
- 후보별 그룹: `"{후보명} {카테고리}"` (예: "김용남 지지율")

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `min_score` | 0.3 | 유사도 임계값 미만 제거 (그룹별 적용) |
| `deduplicate` | true | 동일 기사 URL 중복 제거 — 그룹 내에서 최고 점수만 유지 |
| `cross_encoder.enabled` | false | Cross-encoder 재정렬 활성화 여부 |
| `cross_encoder.model` | `BAAI/bge-reranker-v2-m3` | 한국어 지원 Cross-encoder (~560MB, 최초 실행 시 다운로드) |
| `cross_encoder.top_n` | 10 | 그룹당 Cross-encoder 재정렬 후 상위 N건 유지 |

### Step 3: Score — 트리 구조 프롬프트 + 여론조사 오차범위 기반 LLM 판정

Reranker를 통과한 그룹별 청크가 **후보 → 분석 항목** 트리 구조로 LLM에 전달됩니다. 카테고리당 **score 상위 3건**만 포함하여 토큰 사용량을 제어합니다.

**System Prompt** (역할 + 판정 기준 + 출력 형식):
```
당신은 한국 선거 판세 분석 전문가입니다.
후보별 9가지 분석 항목을 도출합니다.

판정 기준 (여론조사 오차범위 ±3%p 적용):
- 여론조사 지지율은 ±3%p의 오차범위를 가짐. 두 후보 간 격차가 6%p 이내이면 통계적 동률
- "우세": 1위 후보와의 격차가 6%p 초과로 선두
- "균형": 1위 후보와의 격차가 6%p 이내 (오차범위 중첩 구간)
- "열세": 1위 후보와의 격차가 6%p 초과로 뒤처짐
- 오차범위 내에서는 기사 논조·추세·이슈 등을 종합하여 최종 판정

규칙:
- 모든 후보의 win_probability 합계는 반드시 1.0
- reasoning 내부의 각 항목은 한국어로 3~5문장씩, 기사 내용에 근거하여 구체적으로 작성

JSON 형식:
{candidates: [{candidate, verdict, win_probability, reasoning}], summary}
```

**reasoning 9가지 분석 항목** (query_templates.json 검색 축과 1:1 매칭):

| 필드 | 라벨 | 설명 |
|------|------|------|
| `support_rate` | 지지율 | 최신 여론조사 수치, 정당 지지율 대비 개인 지지율, 타 후보와의 격차 |
| `pledge_reaction` | 공약 반응 | 핵심 공약에 대한 유권자·언론 반응, 실현 가능성, 공약 차별성 |
| `strengths` | 강점 | 유리한 요인 (조직력, 인지도, 지지 기반, 긍정 보도 등) |
| `weaknesses` | 약점 | 불리한 요인 (부정 보도, 내부 갈등, 약한 인지도, 리스크 등) |
| `issues` | 이슈 | 주요 이슈·논란·쟁점 (스캔들, 정책 논쟁, 당내 갈등 등) |
| `support_trend` | 지지율 추이 | 시간 경과에 따른 지지율 변화 방향, 변곡점과 원인 |
| `public_opinion` | 출마 여론 | 출마에 대한 여론 반응, 지역구 민심, 당내·당외 지지 수준 |
| `strategy` | 선거 전략 | 지지율을 끌어올리기 위한 구체적 전략 (5~7문장) |
| `forecast` | 예측 | 향후 판세 전망, 추세 변화, 변수, 당선 가능성 근거 |

**User Prompt** (트리 구조 — 후보별 × 분석 항목별 그룹핑):
```
## 선거구: 부산북구갑

## 후보 목록
- 하정우 (더불어민주당)
- 한동훈 (무소속)
- 박민식 (국민의힘)

## 최신 여론조사 (오차범위 ±3%p)
조사기관: JTBC, 조사일: 2026-05-07
- 하정우 (더불어민주당): 37% (오차범위: 34.0%~40.0%)
- 한동훈 (무소속): 26% (오차범위: 23.0%~29.0%)
- 박민식 (국민의힘): 25% (오차범위: 22.0%~28.0%)
※ 두 후보 간 격차 6%p 이내는 통계적 동률 (오차범위 중첩)

## 공통 판세 자료

### 판세 (상위 3/12건)
[1] 부산 북구갑 3파전 양상 (naver_news, 2026-05-09) — score: 0.892
    기사 본문 200자 미리보기...

### 여론조사 (상위 3/8건)
[1] 부산 북구갑 여론조사 결과 발표 (naver_news, 2026-05-08) — score: 0.871
    기사 본문 200자 미리보기...

## 후보별 분석 자료

### 하정우 (더불어민주당)

#### 지지율 (상위 3/15건)
[1] 하정우 지지율 상승세 (naver_news, 2026-05-08) — score: 0.845
    기사 본문 200자 미리보기...

#### 공약 반응 (상위 3/7건)
[1] 하정우 교육 공약 반응 (naver_news, 2026-05-07) — score: 0.812
    ...

#### 강점 (상위 2/2건)
...
#### 약점 (상위 1/1건)
...
(이하 이슈, 지지율 추이, 출마 여론, 선거 전략 동일 구조)

### 한동훈 (무소속)

#### 지지율 (상위 3/10건)
...
(이하 동일 구조)

## 요청
위 분석 항목별로 분류된 근거를 참고하여 각 후보의 verdict, win_probability,
reasoning을 JSON으로 출력하세요.
각 reasoning 필드(support_rate, pledge_reaction, strengths 등)는 해당 분석 항목에
배치된 근거 기사를 우선적으로 참고하여 구체적으로 작성하세요.
```

LLM은 **질의어와 함께 분류된 근거 기사**를 보고 각 분석 항목을 작성합니다. 카테고리당 score 상위 3건 × 200자 미리보기로 토큰 사용량을 제어하며 (약 15,000~18,000 토큰), 여론조사 데이터에는 ±3%p 오차범위가 명시되어 LLM이 접전 구간을 통계적으로 판단합니다. 응답이 JSON으로 파싱된 후 확률 합이 1.0이 아니면 자동 정규화됩니다.

**CLI 출력 형식** (9가지 항목 개별 표시):
```
  🔴 김용남 (더불어민주당)
     판정: 우세  |  승률: 35.0%
     ██████████░░░░░░░░░░░░░░░░░░░░
     📊 지지율: 최신 여론조사에서 28.3%로 1위...
     📋 공약 반응: 평택 교통 인프라 공약에 대해 긍정적 반응...
     💪 강점: 지역 인지도 높고 당 조직력 탄탄...
     ⚠️ 약점: 젊은층 인지도 부족, 정책 차별화 미흡...
     🔥 이슈: 부동산 개발 관련 논란이 일부 제기...
     📈 지지율 추이: 최근 2주간 3%p 상승세 유지...
     🗳️ 출마 여론: 지역구 민심은 대체로 우호적...
     🎯 선거 전략: 중도층 공략을 위해 비당파적 지역 현안 강조...
     🔮 예측: 현재 추세 유지 시 당선 가능성 높으나...
```

### Step 4: Save — 판정 결과 저장

판정 결과는 `VerdictStore`를 통해 `data/verdicts/{district_id}.jsonl`에 누적 저장되며, API를 통해 최신/이력/시계열 데이터로 조회할 수 있습니다.

### Scorer 구현체

| Scorer | config 키 | 모델 | 특징 |
|--------|----------|------|------|
| OpenAIScorer | `openai` | GPT-4o | **기본값**, `json_object` 응답 모드 |
| AnthropicScorer | `anthropic` | Claude | API 키 추가 시 전환 가능 |

### RAG 로깅 정책

| 컴포넌트 | WARNING | INFO | DEBUG |
|---------|---------|------|-------|
| Retriever | 검색 결과 변환 실패 | 쿼리 생성, 임베딩 완료, VectorDB 검색 건수, 시간 필터, 개별 결과 (score/title/source), 후보별 신규/중복 건수 | - |
| Reranker | 빈 입력 | 재정렬 완료 (전후 건수, cross_encoder 유무) | 임계값/중복 제거 건수, Cross-encoder 재정렬 건수 |
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
| `GET` | `/api/v1/admin/polls` | 여론조사 목록 조회 (`?district_id=` 필터 가능) |
| `POST` | `/api/v1/admin/polls` | 여론조사 일괄 저장 (조사기관·날짜·후보별 지지율) |
| `DELETE` | `/api/v1/admin/polls/{id}` | 여론조사 개별 삭제 |
| `DELETE` | `/api/v1/admin/polls` | 여론조사 전체 삭제 (`?district_id=` 필터 가능) |

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

# 여론조사 입력 (부산북구갑, 한국갤럽 2026-05-05)
curl -X POST http://localhost:8000/api/v1/admin/polls \
  -H "Content-Type: application/json" \
  -d '{"entries": [
    {"district_id": "busan_bukgu_gap", "candidate": "한동훈", "party": "무소속", "support": 38.2, "pollster": "한국갤럽", "survey_date": "2026-05-05"},
    {"district_id": "busan_bukgu_gap", "candidate": "하정우", "party": "더불어민주당", "support": 31.5, "pollster": "한국갤럽", "survey_date": "2026-05-05"},
    {"district_id": "busan_bukgu_gap", "candidate": "박민식", "party": "국민의힘", "support": 22.1, "pollster": "한국갤럽", "survey_date": "2026-05-05"}
  ]}'

# 여론조사 조회
curl http://localhost:8000/api/v1/admin/polls?district_id=busan_bukgu_gap
```

---

## 현재 구현 상태

- [x] 프로젝트 구조 설계, Strategy + Registry 패턴
- [x] Scraper 구현 완료 (NaverNewsScraper, NaverElectionScraper, PoliticalNewsScraper, URL 영속 저장소)
- [x] Chunker 구현 완료 (6종: korean_paragraph, sentence, token, semantic, recursive, article_aware)
- [x] Embedder 구현 완료 (3종: openai, bge_m3, ko_simcse)
- [x] 기사 → 후보/선거구 자동 태깅 (키워드 매칭 + 문맥 검증)
- [x] IngestionPipeline 연결 (scrape→tag→chunk→embed→store)
- [x] VectorDB Repository 구현 완료 (7종: qdrant, chroma, milvus_lite, lancedb, weaviate, pgvector, pinecone)
- [x] VectorDB 안전장치 (결정적 ID, 시간 필터, 만료 정리)
- [x] RAG 판정 엔진 구현 완료 (Retriever, Reranker, Scorer)
- [x] OpenAIScorer (GPT-4o) + AnthropicScorer (Claude)
- [x] 판정 결과 영속 저장 (VerdictStore — 선거구별 JSONL 이력 누적)
- [x] FastAPI 관리자 API (파이프라인 실행, VectorDB 관리, 설정 변경)
- [x] 판세 결과 API (최신/이력/시계열 조회, 판정 실행)
- [x] APScheduler 연동 (cron 주기 자동 수집 + 판정)
- [x] 여론조사 관리 (PollStore — JSONL 이력 저장, 관리자 페이지 스프레드시트형 UI, API CRUD)
- [x] 여론조사 → LLM 판정 연동 (최신 조사 데이터 자동 프롬프트 주입)
- [x] 여론조사 오차범위 ±3%p 적용 (6%p 이내 격차 = 통계적 동률 판정)
- [x] RAG 트리 구조 프롬프트 (후보별 × 분석 항목별 그룹핑된 근거 기사 전달)
- [x] Cross-encoder 재정렬 (bi-encoder 필터 후 BAAI/bge-reranker-v2-m3로 정밀 재평가, config.yaml에서 on/off)
- [x] NaverElectionScraper (네이버 선거 전용 페이지 크롤링)
- [x] 테스트 262개 passed, 15개 skipped
- [x] TypeScript 대시보드 (Next.js 16 + Recharts)
- [x] CORS 미들웨어 (프론트엔드 → 백엔드 API 연동, `CORS_ORIGINS` 환경변수 지원)
- [x] Docker 배포 (백엔드 Dockerfile + 프론트엔드 멀티스테이지 Dockerfile + docker-compose.yml)
