# Election Radar — 개발 이력 (포트폴리오)

## 프로젝트 개요

**Election Radar** — 2026년 재보궐선거(평택을·부산북구갑) 판세를 실시간 분석하는 AI 기반 웹 서비스

뉴스 자동 크롤링 → 텍스트 청킹 → 벡터 임베딩 → VectorDB 저장 → RAG 기반 판세 판정 → 대시보드 시각화까지 이어지는 **End-to-End 파이프라인**을 단독 설계·구현.

| 항목 | 내용 |
|------|------|
| 기간 | 2026.04 ~ |
| 역할 | 기획·설계·개발·배포 전 과정 (1인 개발) |
| 규모 | 백엔드 Python 80파일 / 6,000+ LoC, 프론트엔드 TypeScript 1,200 LoC, 테스트 344개 |

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 백엔드 | Python 3.11, FastAPI, APScheduler, LangGraph |
| LLM/AI | OpenAI GPT-4o, Anthropic Claude, RAG (자체 구현) |
| 임베딩 | OpenAI text-embedding-3-small, BAAI/bge-m3, KoSimCSE |
| 재정렬 | BAAI/bge-reranker-v2-m3 (Cross-encoder) |
| Vector DB | ChromaDB (개발), Qdrant (운영) 외 7종 지원 |
| 한국어 NLP | kss (문장 분리), kiwipiepy |
| 프론트엔드 | Next.js 16, TypeScript, Tailwind CSS, Recharts |
| 인프라 | Docker Compose, 멀티스테이지 빌드 |
| 패키지 관리 | uv (백엔드 — Rust 기반 초고속 Python 패키지 매니저), npm (프론트엔드 — Node.js 표준 패키지 매니저) |

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
- **LangGraph 멀티스테이지 오케스트레이션**: 5종 분석 모드(판정·진단·전략·비교·여론조사) + 역할별 프롬프트 분리 + 3중 검증(근거 일치·일관성·확률 범위) → 실패 시 자동 재판정 (최대 2회). Google Sheets 여론조사 연동 및 조사 방법론별 신뢰도 분석 포함. 노드별 구조화 JSONL 로깅으로 전 과정 추적 가능

### 2. LangChain 없이 자체 구현한 이유

LangChain은 범용 RAG를 빠르게 프로토타이핑할 수 있는 프레임워크이지만, 본 프로젝트의 도메인 특화 요구사항에는 자체 구현이 더 적합하다고 판단.

**LangChain 사용 시 (5줄이면 기본 RAG 완성)**
```python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(splitter.split_documents(loader.load()), OpenAIEmbeddings())
chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
```

**자체 구현을 선택한 이유**

| 항목 | LangChain | 자체 구현 (본 프로젝트) |
|------|-----------|----------------------|
| 데이터 모델 | `Document(page_content, metadata)` 단일 모델 | `RawArticle → Article → Chunk → ChunkWithEmbedding → SearchResult → CandidateScore → DailyVerdict` 7단계 도메인 모델 |
| 태깅 | 메타데이터 수동 관리 | 키워드 매칭 + ±50자 문맥 검증으로 후보/선거구 자동 태깅 |
| 검색 | 단일 쿼리 → top-k | 후보 × 분석 항목(8축) 그룹별 쿼리 재구성 + Cross-encoder 재정렬 |
| 판정 | 단일 프롬프트 체인 | 5종 역할별 프롬프트 + LangGraph 멀티스테이지 (score → validate → correct → diagnose) |
| 여론조사 | 지원 없음 | Google Sheets 2시트 FK JOIN + 오차범위 ±3%p 통계적 판정 |
| 디버깅 | 추상화 레이어가 깊어 추적 어려움 | 코드 직접 추적 가능, 노드별 구조화 로깅 |
| 의존성 | `langchain` + 수십 개 하위 패키지 | 필요한 것만 (`openai`, `chromadb`, `langgraph`) |

**결론**: LangChain의 `Document → TextSplitter → VectorStore → Retriever → Chain` 규격 안에서는 한국어 선거 도메인 특화 로직(문맥 검증 태깅, 후보별 그룹 검색, 오차범위 반영 판정, 여론조사 시계열 분석)을 구현하기 오히려 복잡해짐. `ComponentRegistry` + Strategy 패턴으로 LangChain과 동등한 교체 유연성을 확보하면서도, 도메인 요구사항에 최적화된 파이프라인을 구축.

### 3. Strategy + Registry 패턴으로 교체 가능한 아키텍처

모든 핵심 컴포넌트(Scraper 3종, Chunker 6종, Embedder 3종, VectorDB 7종, Scorer 2종, PollStore 2종)를 **설정 파일(config.yaml) 변경만으로 전환** 가능하도록 설계.

```yaml
# 예: VectorDB를 ChromaDB에서 Qdrant로 전환 — 코드 수정 불필요
vectordb:
  type: qdrant    # chroma → qdrant
```

- `ComponentRegistry` 기반 자동 등록 데코레이터
- Lazy Import 패턴: 미사용 구현체의 패키지 미설치에도 다른 컴포넌트 정상 동작
- 새 구현체 추가 시 기존 코드 변경 없이 확장 가능 (OCP)

### 4. 2단계 재정렬 (Bi-encoder + Cross-encoder)

VectorDB의 코사인 유사도(bi-encoder)만으로는 "강점" vs "약점"처럼 의미적으로 가까운 개념의 구분이 어려운 한계를 **Cross-encoder(BAAI/bge-reranker-v2-m3)로 보완**.

```
Stage 1: bi-encoder score → 임계값 필터링 + URL 중복 제거 (빠른 후보 추출)
Stage 2: Cross-encoder → (query, chunk) 쌍 정밀 재평가 (상위 N건만, sigmoid 정규화)
```

- 그룹별(후보 × 분석 항목) 쿼리 재구성 후 Cross-encoder 적용
- config.yaml에서 on/off, 모델, top_n 설정 가능

### 5. RAG 파이프라인 실행 절차

실제 쿼리 `"부산 북구갑 여론조사 추이는?"` 실행 시 4단계로 처리:

```
[1단계] 컴포넌트 초기화
        embedder(OpenAI text-embedding-3-small) → VectorDB(ChromaDB, 1820건)

[2단계] 검색 + 2단계 재정렬
        쿼리 임베딩 → VectorDB 코사인 유사도 top-20 검색
        → Cross-encoder(bge-reranker-v2-m3) 정밀 재평가 → 20건 → 10건

[3단계] 여론조사 데이터 로드 (쿼리 키워드 자동 감지)
        "여론조사" 키워드 + "북구갑" → 부산북구갑 선거구 매칭
        → Google Sheets에서 PollEntry 48건 로드 → 부산북구갑 18건 필터
        → 시계열 표 + ASCII 추이 차트 포맷

[4단계] LLM 답변 생성
        뉴스 청크 10건 + 여론조사 시계열 표 → 프롬프트 조합 → GPT-4o → 답변
```

위 `--query` 모드는 단발 질의 응답으로, LangGraph를 사용하지 않는다. LangGraph는 아래 `--district` 모드의 공식 판정에서만 작동.

### 6. LangGraph 멀티스테이지 오케스트레이션

LLM 판정을 한 번 호출로 끝내지 않고, **LangGraph의 `StateGraph`로 단계별 실행 + 검증 + 자동 재시도**를 선언적으로 오케스트레이션. 단일 프롬프트에 판정·분석·전략을 모두 요구하면 각 항목의 깊이가 얕아지는 한계를 **역할별 프롬프트 분리**로 해결.

**5종 분석 모드별 그래프 플로우:**

```
verdict:       score → validate ──→ END
diagnosis:     score → validate ──→ diagnose → END
strategy:      score → validate ──→ diagnose → strategize → END
comparison:    score → validate ──→ compare → END
opinion_polls: analyze_polls ─────→ END

                       ↑ validate 실패 시
                       └── correct → score (최대 2회 재시도)
```

**각 노드 역할:**

| 노드 | 역할 | 프롬프트 |
|------|------|---------|
| `score` | 뉴스 청크 + 여론조사 데이터를 종합하여 후보별 판세 판정 (verdict, win_probability) | `VERDICT_PROMPT` |
| `validate` | 3중 검증 — ①근거 일치 (근거 0건인데 "우세" 판정 차단) ②일관성 (직전 판정 대비 30%p 급변 감지) ③확률 합계=1.0 | — (규칙 기반) |
| `correct` | 검증 실패 시 상태 초기화 → `score`로 재진입 (최대 2회) | — |
| `diagnose` | 검증 통과된 판정을 입력으로 후보별 문제점·근본 원인 진단 | `DIAGNOSIS_PROMPT` |
| `strategize` | 진단 결과를 입력으로 후보별 대응 전략·실행 계획 도출 | `STRATEGY_PROMPT` |
| `compare` | 지정된 두 후보를 6차원(지지율·공약·조직력·언론·강약점·지역기반) 비교 | `COMPARISON_PROMPT` |
| `analyze_polls` | Google Sheets 여론조사 데이터 기반 조사 방법론(무선전화면접 vs ARS)별 신뢰도 분석 + 추세 진단 | `OPINION_POLLS_PROMPT` |

**핵심 설계:**

- **validate → correct 루프**: LLM이 근거 없이 "우세" 판정하거나, 직전 대비 승률 30%p 이상 급변 시 자동 재판정. 프레임워크 없이 구현하면 `while retry < 2: if not valid: retry` 같은 절차적 코드가 되는데, LangGraph의 `StateGraph` + 조건부 엣지(`add_conditional_edges`)로 선언적으로 표현
- **score → diagnose 파이프라인**: score의 검증된 판정 결과가 diagnose의 입력 컨텍스트로 자동 전달되어 일관된 심층 분석 보장
- **opinion_polls 독립 경로**: 뉴스 없이 여론조사 데이터만으로 분석 가능. Google Sheets 2시트 FK JOIN으로 실시간 데이터 로드
- 트리 구조 프롬프트: 후보 → 분석 항목(8축) 구조로 조직화, 카테고리당 상위 3건 × 200자 미리보기로 토큰 예산 관리
- 여론조사 ±3%p 오차범위 적용 (6%p 이내 격차 = 통계적 동률 판정)
- 노드별 구조화 JSONL 로깅으로 전 과정 추적 가능

### 7. VectorDB 안전장치 3중 설계

매번 재생성하지 않고 **누적 업데이트(upsert)** 하되, 3가지 안전장치로 데이터 품질 유지:

| 안전장치 | 방식 |
|---------|------|
| 결정적 ID | `sha256(article_url + chunk_index)` — 동일 기사 중복 벡터 원천 차단 |
| 시간 필터 | `lookback_days` — 검색 시 최근 N일 기사만 사용 |
| 만료 정리 | `purge_days` — 오래된 벡터 주기적 물리 삭제 |

### 8. 다채널 뉴스 크롤링 + 기사 전문 수집 + 자동 태깅

- NaverNewsScraper: SDS 디자인 시스템 기반 동적 셀렉터로 네이버 뉴스 파싱
- NaverElectionScraper: 네이버 선거 전용 페이지 크롤링
- PoliticalNewsScraper: 오마이뉴스·프레시안·미디어오늘 RSS 파싱
- **기사 전문 수집**: 검색 결과 snippet 대신 상세 페이지에서 본문 전문을 추출 (11개 CSS 셀렉터 + og:description fallback). 기사당 평균 157자 → 1,200자 이상으로 정보량 대폭 증가
- **품질 필터**: 50자 미만 극소 청크 및 선거구 미태깅 청크를 VectorDB 저장 전 제거
- 자동 태깅 + 문맥 검증: 기사 제목+본문에서 키워드 매칭 시 **주변 ±50자 윈도우에 선거 맥락 단어 존재 여부를 검증**하여 동음이의어 오태깅 방지 (예: "조국"(정치인) vs "조국"(motherland))
- URL 영속 저장소(`scraped_urls.jsonl`)로 중복 수집 방지

### 9. 대시보드 + 관리자 페이지 (Full-stack)

- **메인 대시보드**: 선거구 탭 선택, 후보별 판정 카드(9가지 분석 항목), 승률 시계열 차트(Recharts)
- **관리자 페이지**: 파이프라인 실행, VectorDB 모니터링, RAG 설정 변경, 여론조사 데이터 입력
- FastAPI RESTful API: 16개 엔드포인트 (CRUD + 파이프라인 제어 + 시계열 조회)
- APScheduler: cron 기반 하루 3회 자동 수집·판정

### 10. Docker 배포 + 운영 준비

- 백엔드: Python 3.11-slim + uv 기반 이미지
- 프론트엔드: Node 20-alpine 멀티스테이지 빌드 (standalone 출력)
- Docker Compose: 수집 데이터·VectorDB 볼륨 영속 저장
- CORS 미들웨어: 환경변수(`CORS_ORIGINS`)로 운영 도메인 설정

---

## 테스트

총 **359개** 테스트 (344 passed, 15 skipped)

| 모듈 | 테스트 수 | 내용 |
|------|----------|------|
| Scraper | 40 | 네이버·정치 매체 크롤링, 기사 전문 수집, URL 영속 저장소 |
| Tagger | 28 | 키워드 매칭 + 문맥 검증 자동 태깅 |
| Chunker | 37 | 6종 청커 (한국어 문단, 문장, 토큰, 의미, 재귀, 기사 구조 인식) |
| Embedder | 16 | 3종 임베더 (OpenAI, BGE-M3, KoSimCSE) |
| Pipeline | 27 | 수집 파이프라인 E2E + 품질 필터 |
| VectorDB | 48 | 7종 VectorDB 구현체 (CRUD, 필터, 만료 정리) |
| RAG | 132 | Retriever 15 + Reranker 14 + Scorer 33 + VerdictGraph 29 + VerdictStore 11 + PollStore 30 |
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

### 동음이의어 오태깅 ("조국" 문제)

**문제**: "카자흐스탄 고려인은 가슴속에 늘 조국을 품고 살아온 핏줄이다" 같은 기사가 정치인 "조국" 후보 관련 기사로 오태깅됨. 한국어에서 인명과 일반 명사가 동일한 표기를 갖는 동음이의어 문제.

**해결**: 키워드 매칭 시 **문맥 검증(Context Validation)** 도입. 매칭된 키워드 주변 ±50자 윈도우 내에 선거 관련 단어(후보, 출마, 선거, 재보궐, 지지율, 정당명 등 38개)가 존재해야 유효한 매칭으로 인정. 추가로 config.yaml에 구체적 키워드("조국 후보", "조국 대표", "조국혁신당")를 확장하여 정밀도 향상.

### 기사 본문 정보량 부족 (snippet 수집 한계)

**문제**: 네이버 뉴스 검색 결과에서 snippet(요약문, 평균 157자)만 수집하여, 청킹 후 VectorDB에 저장되는 정보량이 절대적으로 부족. 기사당 1~2개 청크만 생성되어 RAG 검색 시 핵심 내용 누락.

**해결**: `fetch_article_body()` 함수를 구현하여 기사 상세 페이지에서 전문 추출. 11개 CSS 셀렉터 순차 시도 + og:description fallback + 불필요 요소(script, style, 저작권 영역) 자동 제거. 기사당 평균 1,200자 이상 확보, 청크 수 6배 이상 증가(54건 → 346건).

### VectorDB 노이즈 유입 (극소 청크 + 미태깅)

**문제**: VectorDB 저장 데이터의 24%가 50자 미만 극소 청크(제목만 남은 잔여물), 24%가 선거구 미태깅 청크(무관한 뉴스). 검색 결과에 불필요한 노이즈 유발.

**해결**: `_filter_chunks()` 품질 필터 도입. 50자 미만 청크 및 district_id 미태깅 청크를 VectorDB 저장 전 제거. 필터 적용 후 VectorDB 전체 재구축하여 노이즈 0% 달성.

### VectorDB 데이터 무결성

**문제**: 반복 수집 시 동일 기사 중복 벡터 축적 → 검색 결과 노이즈 증가

**해결**: `sha256(article_url + chunk_index)` 결정적 ID로 upsert 시 자동 덮어쓰기. 추가로 시간 필터 + 만료 정리 3중 안전장치 적용.

---

## 기술적 성과 요약

- **End-to-End RAG 시스템**: 데이터 수집부터 LLM 판정, 대시보드 시각화까지 전 과정을 1인 설계·구현
- **교체 가능한 컴포넌트 아키텍처**: 23종의 구현체(Scraper 3 + Chunker 6 + Embedder 3 + VectorDB 7 + Scorer 2 + PollStore 2)를 설정 파일만으로 전환
- **2단계 Reranking**: bi-encoder + Cross-encoder 파이프라인으로 검색 정확도 향상
- **LangGraph 멀티스테이지 오케스트레이션**: 5종 분석 모드(판정·진단·전략·비교·여론조사) + 역할별 프롬프트 분리 + 3중 검증 + 자동 재판정. 구조화 JSONL 로깅으로 노드별 실행 이력 추적
- **Google Sheets 여론조사 연동**: survey_id FK 기반 2시트 JOIN으로 실시간 여론조사 데이터 로드 + 조사 방법론(무선전화면접 vs ARS)별 신뢰도 분석
- **프롬프트 엔지니어링**: 5종 역할별 프롬프트(VERDICT / DIAGNOSIS / STRATEGY / COMPARISON / OPINION_POLLS) + 트리 구조 + 토큰 예산 관리 + 오차범위 기반 통계적 판정 설계
- **Full-stack 개발**: Python(FastAPI) 백엔드 + TypeScript(Next.js) 프론트엔드 + Docker 배포
- **데이터 품질 파이프라인**: 기사 전문 수집(11개 셀렉터) + 기사 구조 인식 청킹(제목 prefix + 리드 분리) + 문맥 검증 태깅(±50자 윈도우) + 품질 필터(50자 미만·미태깅 제거)
- **품질 관리**: 344개 테스트, Pydantic v2 도메인 모델, 3단계 로깅 정책
