---
name: rag
description: >
  Election Radar 프로젝트의 RAG 판정 엔진 컴포넌트를 구현하거나 수정할 때 사용.
  Retriever, Reranker, Scorer 추가·변경, 판정 로직(우세/균형/열세) 수정,
  승리 확률 산출 알고리즘 변경, RAG 설정(config.yaml) 변경, RAG 테스트 작성 시
  반드시 이 파일을 먼저 읽으세요.
  VectorDB에서 검색된 결과를 입력으로 받아 후보별 판정과 승리 확률을 산출합니다.
---

# RAG 판정 엔진 가이드

## 역할과 데이터 흐름

VectorDB에 저장된 벡터를 검색하여 후보별 판세를 판정하고
승리 확률을 산출합니다. 최종 결과는 API를 통해 프론트엔드 대시보드에 제공됩니다.

```
사용자 질의 / 스케줄 트리거
        ↓
Embedder.embed_query(query)            ← 질의 텍스트를 벡터로 변환
        ↓
Retriever.retrieve(query_vector)       ← VectorDB에서 관련 청크 검색 + 시간 필터
        ↓
Reranker.rerank(query, results)        ← 임계값 필터링 + URL 중복 제거 + 점수 정렬
        ↓
Scorer.score(chunks, district)         ← LLM 기반 판정 + 확률 정규화
        ↓
CandidateScore / DailyVerdict          ← API 응답 → 대시보드 시각화
```

**RAG는 판정만** 담당합니다.
뉴스 수집은 Scraper, 벡터 변환은 Embedder, 벡터 저장은 VectorDB가 처리합니다.

---

## 파일 구조

```
rag/
├── pipeline.py          ← RAG 파이프라인 CLI (retrieve → rerank → score)
├── retriever.py         ← Retriever (VectorDB 검색 + 필터링 + 시간 필터)
├── reranker.py          ← Reranker (임계값 필터링 + URL 중복 제거 + 정렬)
├── scorer.py            ← AbstractScorer ABC + ScorerRegistry + 프롬프트 + 파싱
├── openai_scorer.py     ← OpenAIScorer (GPT-4o, json_object 모드)
├── anthropic_scorer.py  ← AnthropicScorer (Claude)
└── __init__.py          ← 구현체 자동 등록을 위한 import
```

> Scorer만 Strategy + Registry 패턴 적용.
> Retriever와 Reranker는 단일 구현체로, 교체 필요 시 Registry 전환 가능.

---

## 도메인 모델

RAG 파이프라인의 입출력을 정의하는 Pydantic 모델입니다.
`models/score.py`에 구현되어 있습니다.

```python
# models/score.py

class SearchResult(BaseModel):
    """VectorDB 검색 결과 단위. Retriever 출력."""
    id: str
    score: float                    # 코사인 유사도 (0~1)
    text: str
    article_url: str
    source: str
    title: str
    published_at: datetime
    candidate: str
    district_id: str


class CandidateScore(BaseModel):
    """후보별 판정 결과. Scorer 출력."""
    candidate: str
    party: str
    district_id: str
    verdict: str                    # "우세" | "균형" | "열세"
    win_probability: float          # 0.0 ~ 1.0
    reasoning: str                  # LLM이 생성한 판정 근거
    supporting_chunks: list[str]    # 근거가 된 청크 ID 목록
    chunk_count: int                # 분석에 사용된 청크 수


class DailyVerdict(BaseModel):
    """선거구별 일일 판정 결과. API 응답 단위."""
    district_id: str
    district_name: str
    date: datetime
    candidates: list[CandidateScore]
    total_chunks_analyzed: int
    summary: str                    # 선거구 전체 요약 (LLM 생성)
```

### 데이터 흐름과 모델 매핑

```
VectorDB.search()  →  list[dict]
        ↓
Retriever          →  list[SearchResult]      ← dict → Pydantic 변환 + 시간 필터
        ↓
Reranker           →  list[SearchResult]      ← 임계값 필터링 + 중복 제거 + 정렬
        ↓
Scorer             →  DailyVerdict            ← LLM 판정 + 확률 정규화
                        ├── candidates: list[CandidateScore]
                        └── summary: str
```

---

## Retriever

### 역할

VectorDB에서 특정 선거구·후보·기간에 해당하는 관련 청크를 검색합니다.

### 핵심 기능

1. **질의 임베딩**: `Embedder.embed_query()`로 질의 텍스트를 벡터 변환
2. **필터 검색**: `district_id`, `candidate` 필터 적용. 필터 검색 0건 시 필터 없이 재검색(fallback)
3. **시간 필터**: `lookback_days` 설정 시 `published_at` 기준 최근 N일 기사만 후처리 필터링
4. **dict → SearchResult 변환**: VectorDB 반환값을 도메인 모델로 정규화

```python
# rag/retriever.py

class Retriever:

    def __init__(self, embedder, vector_repo, top_k=20, lookback_days=None):
        ...

    def retrieve(self, query, district_id, candidate=None) -> list[SearchResult]:
        # 1. embed_query → 2. VectorDB search (필터) → 3. fallback → 4. 시간 필터
        ...

    def retrieve_for_district(self, district) -> list[SearchResult]:
        # 모든 후보에 대해 retrieve() 호출 → ID 기반 중복 제거 → 통합 결과
        ...
```

### 검색 전략

| 전략 | 설명 |
|------|------|
| **후보별 검색** | 각 후보 이름으로 개별 검색 → 후보당 top_k개 |
| **필터 fallback** | `district_id + candidate` 필터 0건 → 필터 없이 재검색 |
| **시간 필터** | `lookback_days` 설정 시 최근 N일 기사만 사용 |

---

## Reranker

### 역할

Retriever가 반환한 검색 결과를 정제합니다.

### 처리 단계

1. **유사도 임계값 필터링**: `min_score` 이하 결과 제거
2. **URL 중복 제거**: 동일 `article_url` → 최고 점수만 유지
3. **점수 내림차순 정렬**

```python
# rag/reranker.py

class Reranker:

    def __init__(self, min_score=0.3, deduplicate=True):
        ...

    def rerank(self, query, results) -> list[SearchResult]:
        # 1. score >= min_score 필터 → 2. URL dedup → 3. 정렬
        ...
```

---

## Scorer

### 역할

Reranker가 정제한 관련 청크를 LLM에 전달하여 후보별 판세를 판정합니다.
Strategy + Registry 패턴 적용으로 `config.yaml`의 `provider` 값으로 OpenAI/Anthropic 전환 가능.

### 핵심 출력

| 필드 | 설명 | 값 범위 |
|------|------|---------|
| `verdict` | 판세 판정 | `"우세"` / `"균형"` / `"열세"` |
| `win_probability` | 승리 확률 | 0.0 ~ 1.0 (전 후보 합계 = 1.0) |
| `reasoning` | 판정 근거 | LLM이 생성한 한국어 2~3문장 |
| `summary` | 선거구 요약 | 전체 판세 한줄 요약 |

### 구현체

| Scorer | Registry 키 | 모델 | 특징 |
|--------|------------|------|------|
| OpenAIScorer | `openai` | GPT-4o | `json_object` 응답 모드, **기본값** |
| AnthropicScorer | `anthropic` | Claude | API 키 추가 시 전환 가능 |

```python
# rag/scorer.py

class AbstractScorer(ABC):

    @abstractmethod
    def _call_llm(self, system: str, user: str) -> str: ...

    def score(self, chunks, district) -> DailyVerdict:
        # 1. 프롬프트 구성 → 2. LLM 호출 → 3. JSON 파싱 → 4. 확률 정규화
        # 빈 입력 또는 파싱 실패 시 균등 확률 배분
        ...

ScorerRegistry = ComponentRegistry(AbstractScorer, "Scorer")
```

### LLM 프롬프트 구조

```
[시스템 프롬프트]
한국 선거 판세 분석 전문가 역할 + 판정 기준 + JSON 출력 형식

[사용자 프롬프트]
## 선거구: {district_name}

## 후보 목록
- {candidate_1} ({party_1})
- {candidate_2} ({party_2})

## 수집된 뉴스 청크 ({N}건)
[1] {title} ({source}, {date}) — score: {score}
    {text preview (300자)}

## 요청
위 뉴스를 종합 분석하여 각 후보의 verdict, win_probability, reasoning을
JSON으로 출력하세요.
```

### 확률 정규화

LLM 출력의 확률 합이 1.0이 아닐 수 있으므로 후처리로 정규화합니다:

```python
def _normalize_probabilities(scores):
    total = sum(s.win_probability for s in scores)
    if total == 0:
        equal = 1.0 / len(scores)  # 균등 배분
    elif abs(total - 1.0) > 0.01:
        logger.warning("확률 합 %.4f ≠ 1.0 — 정규화 수행", total)
        for s in scores:
            s.win_probability = round(s.win_probability / total, 4)
```

---

## config.yaml RAG 설정

```yaml
rag:
  retriever:
    top_k: 20                        # 후보당 검색 수
    lookback_days: 14                # 최근 N일 기사만 검색 (null이면 전체)

  reranker:
    min_score: 0.3                   # 유사도 임계값
    deduplicate: true                # 동일 기사 중복 제거

  scorer:
    provider: openai                 # openai | anthropic
    model: gpt-4o                    # LLM 모델
    temperature: 0.1                 # 판정 일관성을 위해 낮은 값
    max_tokens: 2000

  purge_days: 60                     # N일 이전 벡터 자동 삭제 (null이면 비활성)
```

---

## CLI 사용법

```bash
# 평택을 판정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b

# 부산북구갑 판정
PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap

# 검색 수 조정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --top-k 10

# 최근 7일 기사만 사용
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --lookback-days 7

# 유사도 임계값 조정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --min-score 0.5

# 검색 결과만 확인 (LLM 판정 생략)
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --skip-score

# 30일 이전 벡터 삭제 후 판정
PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --purge-days 30
```

---

## 로깅 규칙

### Retriever

| 레벨 | 내용 |
|------|------|
| `WARNING` | 검색 결과 변환 실패 (스킵) |
| `INFO` | 필터 fallback 재검색, 시간 필터 적용 (전후 건수), 검색 완료 (건수), 선거구 통합 검색 완료 |
| `DEBUG` | 개별 검색 결과 (id, score, title, published_at) |

### Reranker

| 레벨 | 내용 |
|------|------|
| `WARNING` | 빈 입력 |
| `INFO` | 재정렬 완료 (전후 건수, min_score, dedup 설정) |
| `DEBUG` | 임계값 필터링 제거 건수, URL 중복 제거 건수 |

### Scorer

| 레벨 | 내용 |
|------|------|
| `WARNING` | 검색 결과 0건 (균등 확률 배분), LLM 응답 파싱 실패, 확률 합 ≠ 1.0 |
| `INFO` | LLM 판정 요청 (선거구, 청크 수), LLM 응답 수신 (소요 시간), 판정 완료 (후보별 verdict + 확률), client 초기화 |
| `DEBUG` | LLM 프롬프트 전문, LLM 응답 전문 |

---

## VectorDB 안전장치

RAG 파이프라인은 기존 VectorDB에 누적 업데이트(upsert) 방식을 사용하며,
다음 세 가지 안전장치로 데이터 품질을 유지합니다:

| 안전장치 | 위치 | 설명 |
|---------|------|------|
| **(a) 결정적 ID** | `models/chunk.py` | `sha256(article_url + chunk_index)` → 중복 벡터 원천 차단 |
| **(b) 시간 필터** | `rag/retriever.py` | `lookback_days` 설정으로 최근 N일 기사만 검색 |
| **(c) 만료 정리** | `vectordb/base.py` | `purge_days` 설정으로 오래된 벡터 물리 삭제 |

---

## 테스트 현황

| 파일 | 테스트 수 | 주요 검증 대상 |
|------|----------|--------------|
| `tests/rag/test_retriever.py` | 15개 | embed_query 호출, 필터 검색, fallback, SearchResult 변환, 시간 필터, 선거구 통합 검색, 중복 제거 |
| `tests/rag/test_reranker.py` | 9개 | 임계값 필터링, URL 중복 제거, 정렬, 통합 동작 |
| `tests/rag/test_scorer.py` | 17개 | 확률 정규화, JSON 파싱, 마크다운 래핑 처리, 프롬프트 구성, LLM mock, 빈 입력, 파싱 실패 |

### mock 패턴

```python
# Retriever 테스트 — embedder + vector_repo mock
embedder = MagicMock()
embedder.embed_query.return_value = [0.1] * 1536
repo = MagicMock()
repo.search.return_value = [{"id": "chunk-001", "score": 0.85, ...}]
retriever = Retriever(embedder=embedder, vector_repo=repo, top_k=10)

# Scorer 테스트 — LLM 응답 mock
class FakeScorer(AbstractScorer):
    def _call_llm(self, system, user):
        return '{"candidates": [...], "summary": "..."}'
```

---

## 새 Scorer 구현 체크리스트

### 1단계: 구현 파일 생성

```python
# rag/my_scorer.py

@ScorerRegistry.register("my_provider")
class MyScorer(AbstractScorer):

    def __init__(self, model="my-model", temperature=0.1, max_tokens=2000):
        ...

    @property
    def name(self) -> str:
        return "my_provider"

    def _call_llm(self, system: str, user: str) -> str:
        # LLM API 호출, 응답 텍스트(JSON 문자열) 반환
        ...
```

### 2단계: `pipeline.py`에 import 추가

```python
import rag.my_scorer  # noqa: F401  ← _build_scorer() 내부에 추가
```

### 3단계: config.yaml 변경

```yaml
rag:
  scorer:
    provider: my_provider
    model: my-model
```

### 4단계: 테스트 작성 (`tests/rag/test_scorer.py` 참조)

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| 확률 합 ≠ 1.0 | `_normalize_probabilities()`가 자동 정규화하지만, LLM 프롬프트에서도 명시 |
| LLM 응답을 파싱 없이 사용 | `_parse_llm_response()`로 JSON 파싱 + 예외 처리 필수 |
| 전체 청크를 LLM에 전달 | top_k + Reranker로 제한 (비용·토큰 관리) |
| Retriever에서 검색 없이 전체 스캔 | VectorDB의 vector search + filters 활용 |
| 판정 근거 없이 확률만 반환 | `reasoning` + `supporting_chunks` 필수 |
| 모듈 최상단에서 `import openai` | `_ensure_client()` 내에서 lazy import |
| `_call_llm()` 대신 `score()` 오버라이드 | `_call_llm()`만 오버라이드. `score()`는 부모의 공통 메서드 |
| `pipeline.py`에 새 Scorer import 누락 | `_build_scorer()` 내에서 반드시 import |

---

## 관련 파일 참조

- 전체 아키텍처: `CLAUDE.md`
- 도메인 모델: `backend/models/score.py`
- VectorDB 가이드: `.claude/backend/vectordb/SKILL.md`
- Embedder 가이드: `.claude/backend/ingestion/embedder/SKILL.md`
- Pipeline Orchestrator: `backend/ingestion/pipeline.py`
- config.yaml: `backend/config/config.yaml`
- API 라우터: `backend/app/api/v1/routes/scores.py` (구현 예정)
- 테스트: `backend/tests/rag/`
