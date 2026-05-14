# Election Radar — 개발 이력 (포트폴리오)

## 프로젝트 개요

**Election Radar** — 2026년 재보궐선거(평택을·부산북구갑) 판세를 실시간 분석하는 AI 기반 웹 서비스

뉴스 자동 크롤링 → 텍스트 청킹 → 벡터 임베딩 → VectorDB 저장 → RAG 기반 판세 판정 → 대시보드 시각화까지 이어지는 **End-to-End 파이프라인**을 단독 설계·구현.

| 항목 | 내용 |
|------|------|
| 기간 | 2026.04 ~ |
| 역할 | 기획·설계·개발·배포 전 과정 (1인 개발) |
| 규모 | 백엔드 Python 78파일 / 5,800+ LoC, 프론트엔드 TypeScript 1,200 LoC, 테스트 244개 |

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 백엔드 | Python 3.11, FastAPI, APScheduler |
| LLM/AI | OpenAI GPT-4o, Anthropic Claude, RAG (자체 구현) |
| 임베딩 | OpenAI text-embedding-3-small, BAAI/bge-m3, KoSimCSE |
| 재정렬 | BAAI/bge-reranker-v2-m3 (Cross-encoder) |
| Vector DB | ChromaDB (개발), Qdrant (운영) 외 7종 지원 |
| 한국어 NLP | kss (문장 분리), kiwipiepy |
| 프론트엔드 | Next.js 16, TypeScript, Tailwind CSS, Recharts |
| 인프라 | Docker Compose, 멀티스테이지 빌드 |
| 패키지 관리 | uv (백엔드), npm (프론트엔드) |

---

## 핵심 기술 특장점

### 1. 완전한 RAG 파이프라인 설계·구현

학술 데모 수준이 아닌, 실시간 데이터 수집부터 사용자 대시보드까지 이어지는 **프로덕션 레벨 End-to-End RAG 시스템**을 직접 설계하고 구현.

```
수집(Scrape) → 태깅(Tag) → 청킹(Chunk) → 임베딩(Embed) → 저장(Store)
→ 검색(Retrieve) → 2단계 재정렬(Rerank) → LLM 판정(Score) → 대시보드(Dashboard)
```

- 수집 파이프라인: 크롤링 → 후보/선거구 자동 태깅 → 청킹 → 임베딩 → VectorDB 저장
- 판정 파이프라인: 의미 검색 → bi-encoder 필터 + Cross-encoder 정밀 재평가 → LLM 판정 + 확률 정규화

### 2. Strategy + Registry 패턴으로 교체 가능한 아키텍처

모든 핵심 컴포넌트(Scraper 3종, Chunker 5종, Embedder 3종, VectorDB 7종, Scorer 2종)를 **설정 파일(config.yaml) 변경만으로 전환** 가능하도록 설계.

```yaml
# 예: VectorDB를 ChromaDB에서 Qdrant로 전환 — 코드 수정 불필요
vectordb:
  type: qdrant    # chroma → qdrant
```

- `ComponentRegistry` 기반 자동 등록 데코레이터
- Lazy Import 패턴: 미사용 구현체의 패키지 미설치에도 다른 컴포넌트 정상 동작
- 새 구현체 추가 시 기존 코드 변경 없이 확장 가능 (OCP)

### 3. 2단계 재정렬 (Bi-encoder + Cross-encoder)

VectorDB의 코사인 유사도(bi-encoder)만으로는 "강점" vs "약점"처럼 의미적으로 가까운 개념의 구분이 어려운 한계를 **Cross-encoder(BAAI/bge-reranker-v2-m3)로 보완**.

```
Stage 1: bi-encoder score → 임계값 필터링 + URL 중복 제거 (빠른 후보 추출)
Stage 2: Cross-encoder → (query, chunk) 쌍 정밀 재평가 (상위 N건만, sigmoid 정규화)
```

- 그룹별(후보 × 분석 항목) 쿼리 재구성 후 Cross-encoder 적용
- config.yaml에서 on/off, 모델, top_n 설정 가능

### 4. 트리 구조 프롬프트 + 여론조사 오차범위 기반 LLM 판정

검색 결과를 **후보 → 분석 항목(8축)** 트리 구조로 조직화하여 LLM에 전달. 단순 flat list 대비 분석 품질 향상 및 토큰 효율 제어.

- 카테고리당 score 상위 3건 × 200자 미리보기로 토큰 예산 관리 (~15,000~18,000 토큰)
- 여론조사 ±3%p 오차범위 적용 (6%p 이내 격차 = 통계적 동률 판정)
- 9가지 분석 항목(지지율, 공약 반응, 강점, 약점, 이슈, 지지율 추이, 출마 여론, 선거 전략, 예측)과 검색 쿼리 축 1:1 매칭

### 5. VectorDB 안전장치 3중 설계

매번 재생성하지 않고 **누적 업데이트(upsert)** 하되, 3가지 안전장치로 데이터 품질 유지:

| 안전장치 | 방식 |
|---------|------|
| 결정적 ID | `sha256(article_url + chunk_index)` — 동일 기사 중복 벡터 원천 차단 |
| 시간 필터 | `lookback_days` — 검색 시 최근 N일 기사만 사용 |
| 만료 정리 | `purge_days` — 오래된 벡터 주기적 물리 삭제 |

### 6. 다채널 뉴스 크롤링 + 자동 태깅

- NaverNewsScraper: SDS 디자인 시스템 기반 동적 셀렉터로 네이버 뉴스 파싱
- NaverElectionScraper: 네이버 선거 전용 페이지 크롤링
- PoliticalNewsScraper: 오마이뉴스·프레시안·미디어오늘 RSS 파싱
- 자동 태깅: 기사 제목+본문에서 config.yaml 후보 키워드 매칭 → `candidate`, `district_id` 자동 부여
- URL 영속 저장소(`scraped_urls.jsonl`)로 중복 수집 방지

### 7. 대시보드 + 관리자 페이지 (Full-stack)

- **메인 대시보드**: 선거구 탭 선택, 후보별 판정 카드(9가지 분석 항목), 승률 시계열 차트(Recharts)
- **관리자 페이지**: 파이프라인 실행, VectorDB 모니터링, RAG 설정 변경, 여론조사 데이터 입력
- FastAPI RESTful API: 16개 엔드포인트 (CRUD + 파이프라인 제어 + 시계열 조회)
- APScheduler: cron 기반 하루 3회 자동 수집·판정

### 8. Docker 배포 + 운영 준비

- 백엔드: Python 3.11-slim + uv 기반 이미지
- 프론트엔드: Node 20-alpine 멀티스테이지 빌드 (standalone 출력)
- Docker Compose: 수집 데이터·VectorDB 볼륨 영속 저장
- CORS 미들웨어: 환경변수(`CORS_ORIGINS`)로 운영 도메인 설정

---

## 테스트

총 **244개** 테스트 (230 passed, 14 skipped)

| 모듈 | 테스트 수 | 내용 |
|------|----------|------|
| Scraper | 33 | 네이버·정치 매체 크롤링, URL 영속 저장소 |
| Tagger | 19 | 키워드 매칭 자동 태깅 |
| Chunker | 27 | 5종 청커 (한국어 문단, 문장, 토큰, 의미, 재귀) |
| Embedder | 16 | 3종 임베더 (OpenAI, BGE-M3, KoSimCSE) |
| Pipeline | 13 | 수집 파이프라인 E2E |
| VectorDB | 48 | 7종 VectorDB 구현체 (CRUD, 필터, 만료 정리) |
| RAG | 57 | Retriever 15 + Reranker 14 + Scorer 17 + VerdictStore 11 |
| API | 31 | Admin 20 + Scores 6 + Scheduler 5 |

외부 의존성(HTTP, DB, LLM)은 전수 mock 처리. 실제 네트워크 호출 없이 테스트 가능.

---

## 문제 해결 사례

### OpenAI 토큰 초과 (429 Rate Limit)

**문제**: 그룹화된 프롬프트 도입 후 요청이 ~53,000 토큰으로 TPM 한도(30,000) 초과

**해결**: 카테고리당 상위 3건 제한 + 미리보기 200자 절삭 → ~15,000~18,000 토큰으로 감축. 정보 손실 없이 score 기반 우선순위로 핵심 근거만 전달.

### bi-encoder의 의미 혼동

**문제**: "김용남 강점" 질의에 "김용남의 약점은..." 청크가 높은 유사도로 검색됨 (임베딩 공간에서 "강점"과 "약점"이 가까이 위치)

**해결**: Cross-encoder(bge-reranker-v2-m3) 2단계 재정렬 도입. 질의+청크를 결합 입력으로 관련성 직접 평가하여 오분류 보정.

### VectorDB 데이터 무결성

**문제**: 반복 수집 시 동일 기사 중복 벡터 축적 → 검색 결과 노이즈 증가

**해결**: `sha256(article_url + chunk_index)` 결정적 ID로 upsert 시 자동 덮어쓰기. 추가로 시간 필터 + 만료 정리 3중 안전장치 적용.

---

## 기술적 성과 요약

- **End-to-End RAG 시스템**: 데이터 수집부터 LLM 판정, 대시보드 시각화까지 전 과정을 1인 설계·구현
- **교체 가능한 컴포넌트 아키텍처**: 20종의 구현체(Scraper 3 + Chunker 5 + Embedder 3 + VectorDB 7 + Scorer 2)를 설정 파일만으로 전환
- **2단계 Reranking**: bi-encoder + Cross-encoder 파이프라인으로 검색 정확도 향상
- **프롬프트 엔지니어링**: 트리 구조 + 토큰 예산 관리 + 오차범위 기반 통계적 판정 설계
- **Full-stack 개발**: Python(FastAPI) 백엔드 + TypeScript(Next.js) 프론트엔드 + Docker 배포
- **품질 관리**: 244개 테스트, Pydantic v2 도메인 모델, 3단계 로깅 정책
