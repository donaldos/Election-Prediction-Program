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
Embedder.embed(query)              ← 질의 텍스트를 벡터로 변환
        ↓
Retriever.retrieve(query_vector)   ← VectorDB에서 관련 청크 검색
        ↓
Reranker.rerank(query, candidates) ← (선택) 검색 결과 재정렬
        ↓
Scorer.score(chunks, district)     ← LLM 기반 판정 + 확률 산출
        ↓
CandidateScore / DailyVerdict      ← API 응답 → 대시보드 시각화
```

**RAG는 판정만** 담당합니다.
뉴스 수집은 Scraper, 벡터 변환은 Embedder, 벡터 저장은 VectorDB가 처리합니다.

---

## 파일 구조 (계획)

```
rag/
├── retriever.py     ← VectorDB 검색 + 필터링 (선거구, 후보, 기간)
├── reranker.py      ← 검색 결과 재정렬 (Cross-encoder 또는 LLM 기반)
└── scorer.py        ← LLM 판정 엔진 (우세/균형/열세 + 승리 확률)
```

> RAG 컴포넌트는 Strategy + Registry 패턴을 **선택적으로** 적용합니다.
> Retriever와 Scorer는 LLM 프로바이더(OpenAI, Claude 등)에 따라 교체 가능하도록 설계하되,
> Reranker는 단순 유틸리티로 시작하여 필요 시 Registry로 전환합니다.

---

## 도메인 모델 (구현 예정)

RAG 파이프라인의 입출력을 정의하는 Pydantic 모델입니다.
`models/score.py`에 작성합니다.

```python
# models/score.py

from pydantic import BaseModel
from datetime import datetime


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
Retriever          →  list[SearchResult]      ← dict → Pydantic 변환
        ↓
Reranker           →  list[SearchResult]      ← 재정렬 (점수 업데이트)
        ↓
Scorer             →  DailyVerdict            ← LLM 판정
                        ├── candidates: list[CandidateScore]
                        └── summary: str
```

---

## Retriever 설계

### 역할

VectorDB에서 특정 선거구·후보·기간에 해당하는 관련 청크를 검색합니다.

### 핵심 기능

1. **질의 임베딩**: 검색 질의를 Embedder로 벡터 변환
2. **필터 검색**: district_id, candidate, published_at 범위로 필터링
3. **dict → SearchResult 변환**: VectorDB 반환값을 도메인 모델로 정규화

```python
# rag/retriever.py (설계안)

class Retriever:

    def __init__(self, embedder, vector_repo, top_k: int = 20):
        self._embedder = embedder
        self._repo = vector_repo
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        district_id: str,
        candidate: str | None = None,
    ) -> list[SearchResult]:
        # 1. 질의 임베딩
        query_vector = self._embedder.embed_query(query)

        # 2. VectorDB 검색 (필터 적용)
        filters = {"district_id": district_id}
        if candidate:
            filters["candidate"] = candidate

        raw_results = self._repo.search(
            query_vector=query_vector,
            top_k=self._top_k,
            filters=filters,
        )

        # 3. SearchResult로 변환
        return [SearchResult(**r) for r in raw_results]
```

### 검색 전략

| 전략 | 설명 | 용도 |
|------|------|------|
| **후보별 검색** | 각 후보 이름으로 개별 검색 → 후보당 top_k개 | 후보별 판세 비교 |
| **선거구 통합 검색** | 선거구 키워드로 통합 검색 → 전체 top_k개 | 선거구 전체 동향 파악 |
| **기간 필터** | published_at 범위로 최근 N일 데이터만 | 최신 판세 반영 |

---

## Reranker 설계

### 역할

Retriever가 반환한 초기 검색 결과를 더 정밀한 관련성 기준으로 재정렬합니다.

### 접근 방식 (우선순위)

1. **점수 기반 필터링** (1단계, 즉시 구현 가능)
   - 유사도 임계값(threshold) 이하 결과 제거
   - 중복 기사 제거 (동일 article_url)

2. **Cross-encoder 재정렬** (2단계, 선택)
   - sentence-transformers의 CrossEncoder로 query-chunk 쌍 점수 재계산
   - 정확도 향상, 하지만 속도 저하

3. **LLM 기반 재정렬** (3단계, 선택)
   - Claude/GPT로 관련성 판단
   - 가장 정확하지만 비용/속도 부담

```python
# rag/reranker.py (설계안)

class Reranker:

    def __init__(self, min_score: float = 0.3):
        self._min_score = min_score

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        # 1. 유사도 임계값 필터링
        filtered = [r for r in results if r.score >= self._min_score]

        # 2. 중복 기사 제거 (같은 article_url → 최고 점수만)
        seen_urls: dict[str, SearchResult] = {}
        for r in filtered:
            if r.article_url not in seen_urls or r.score > seen_urls[r.article_url].score:
                seen_urls[r.article_url] = r

        # 3. 점수 내림차순 정렬
        return sorted(seen_urls.values(), key=lambda r: r.score, reverse=True)
```

---

## Scorer 설계

### 역할

Reranker가 정제한 관련 청크를 LLM에 전달하여 후보별 판세를 판정합니다.

### 핵심 출력

| 필드 | 설명 | 값 범위 |
|------|------|---------|
| `verdict` | 판세 판정 | `"우세"` / `"균형"` / `"열세"` |
| `win_probability` | 승리 확률 | 0.0 ~ 1.0 (전 후보 합계 = 1.0) |
| `reasoning` | 판정 근거 | LLM이 생성한 자연어 설명 |
| `summary` | 선거구 요약 | 전체 판세 한줄 요약 |

### LLM 프롬프트 전략

```python
# rag/scorer.py (설계안)

SYSTEM_PROMPT = """
당신은 한국 선거 판세 분석 전문가입니다.
제공된 뉴스 기사 청크를 분석하여 각 후보의 판세를 판정합니다.

판정 기준:
- "우세": 여론조사 선두, 긍정적 언론 보도 다수, 지지 기반 강화 징후
- "균형": 오차범위 내 접전, 혼재된 신호
- "열세": 여론조사 하위, 부정적 보도, 지지 기반 약화 징후

승리 확률:
- 모든 후보의 확률 합계는 반드시 1.0
- 여론조사 수치가 있으면 이를 기반으로, 없으면 기사 논조를 종합

응답 형식: JSON
"""

class Scorer:

    def __init__(self, llm_client, model: str = "claude-sonnet-4-6"):
        self._llm = llm_client
        self._model = model

    def score(
        self,
        chunks: list[SearchResult],
        district: dict,             # config.yaml의 district 정보
    ) -> DailyVerdict:
        # 1. 후보별 청크 그룹화
        by_candidate = self._group_by_candidate(chunks)

        # 2. LLM에 판정 요청
        prompt = self._build_prompt(by_candidate, district)
        response = self._llm.generate(prompt, system=SYSTEM_PROMPT)

        # 3. JSON 파싱 → CandidateScore 리스트
        scores = self._parse_response(response, district)

        # 4. DailyVerdict 조립
        return DailyVerdict(
            district_id=district["id"],
            district_name=district["name"],
            date=datetime.now(),
            candidates=scores,
            total_chunks_analyzed=len(chunks),
            summary=self._extract_summary(response),
        )
```

### 프롬프트 구조

```
[시스템 프롬프트]
한국 선거 판세 분석 전문가 역할 + 판정 기준 + 출력 형식

[사용자 프롬프트]
## 선거구: {district_name}

## 후보 목록
- {candidate_1} ({party_1})
- {candidate_2} ({party_2})
...

## 수집된 뉴스 청크

### {candidate_1} 관련 ({N}건)
[1] {title} ({source}, {date}) — score: {score}
    {text preview...}

[2] ...

### {candidate_2} 관련 ({N}건)
...

## 요청
위 뉴스를 종합 분석하여 각 후보의 verdict, win_probability, reasoning을 JSON으로 출력하세요.
```

### 확률 정규화

LLM 출력의 확률 합이 1.0이 아닐 수 있으므로 후처리로 정규화합니다:

```python
def _normalize_probabilities(self, scores: list[CandidateScore]) -> list[CandidateScore]:
    total = sum(s.win_probability for s in scores)
    if total == 0:
        equal = 1.0 / len(scores)
        for s in scores:
            s.win_probability = equal
    else:
        for s in scores:
            s.win_probability = s.win_probability / total
    return scores
```

---

## config.yaml 확장 (계획)

```yaml
# RAG 판정 엔진 설정
rag:
  retriever:
    top_k: 20                        # 후보당 검색 수
    lookback_days: 7                 # 최근 N일 데이터만 검색

  reranker:
    min_score: 0.3                   # 유사도 임계값
    deduplicate: true                # 동일 기사 중복 제거

  scorer:
    provider: anthropic              # anthropic | openai
    model: claude-sonnet-4-6         # LLM 모델
    temperature: 0.1                 # 판정 일관성을 위해 낮은 값
    max_tokens: 2000
```

---

## 실행 흐름 (파이프라인 연동)

### 1. 수집 파이프라인 (기존)

```
scrape → chunk → embed → store (VectorDB)
```

이 파이프라인은 하루 3회 스케줄 실행 (07:00, 12:00, 18:00).

### 2. 판정 파이프라인 (신규)

```
retrieve → rerank → score → API 저장
```

수집 파이프라인 완료 후 트리거되거나, 별도 스케줄로 실행.

### 3. API 제공

```
GET /api/v1/scores/{district_id}           ← 최신 판정 결과
GET /api/v1/scores/{district_id}/history   ← 시계열 판정 이력
```

---

## 판정 품질 보장

### 1. 근거 추적성

모든 판정에는 `supporting_chunks` (근거 청크 ID 목록)가 포함됩니다.
대시보드에서 "왜 이 판정인가?"를 클릭하면 원본 기사로 역추적 가능.

### 2. 확률 일관성

- 동일 선거구 전 후보 확률 합 = 1.0
- LLM 출력 후 반드시 정규화

### 3. 시계열 안정성

- 동일 데이터로 재실행해도 유사한 결과 (temperature=0.1)
- 급격한 확률 변동 시 경고 로그

---

## 로깅 규칙

| 레벨 | 내용 |
|------|------|
| `WARNING` | 검색 결과 0건, LLM 응답 파싱 실패, 확률 합 ≠ 1.0 |
| `INFO` | 검색 시작/완료 (청크 수), 판정 시작/완료 (선거구, 후보 수), LLM 호출 |
| `DEBUG` | 개별 청크 점수, LLM 프롬프트/응답 전문, 확률 정규화 전후 |

---

## 테스트 전략

| 대상 | 방식 | mock 대상 |
|------|------|----------|
| Retriever | VectorDB mock → SearchResult 변환 검증 | `vector_repo.search()` |
| Reranker | 필터링·중복제거·정렬 로직 단위 테스트 | 없음 (순수 로직) |
| Scorer | LLM mock → 프롬프트 구조, JSON 파싱, 확률 정규화 검증 | `llm_client.generate()` |
| 통합 | 전체 파이프라인 (retrieve → rerank → score) | VectorDB + LLM 모두 mock |

```python
# tests/rag/test_scorer.py (예시)

def test_probabilities_sum_to_one():
    """모든 후보 확률의 합이 1.0인지 검증."""
    ...

def test_verdict_values():
    """verdict가 '우세', '균형', '열세' 중 하나인지 검증."""
    ...

def test_empty_chunks_returns_equal_probability():
    """검색 결과 0건 시 모든 후보에 균등 확률 배분."""
    ...
```

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| 확률 합 ≠ 1.0 | Scorer에서 반드시 정규화 후처리 |
| LLM 응답을 파싱 없이 사용 | JSON 파싱 + 예외 처리 필수 |
| 전체 청크를 LLM에 전달 | top_k로 제한 (비용·토큰 관리) |
| Retriever에서 검색 없이 전체 스캔 | VectorDB의 vector search + filters 활용 |
| 판정 근거 없이 확률만 반환 | `reasoning` + `supporting_chunks` 필수 |
| 여론조사와 기사 논조 혼동 | 프롬프트에 데이터 유형 명시 |
| 시계열에서 급변동 무시 | 이전 판정 대비 변동폭 경고 로그 |

---

## 관련 파일 참조

- 전체 아키텍처: `CLAUDE.md`
- 도메인 모델: `backend/models/score.py` (구현 예정)
- VectorDB 가이드: `.claude/backend/vectordb/SKILL.md`
- Embedder 가이드: `.claude/backend/ingestion/embedder/SKILL.md`
- Pipeline Orchestrator: `backend/ingestion/pipeline.py`
- config.yaml: `backend/config/config.yaml`
- API 라우터: `backend/app/api/v1/routes/scores.py` (구현 예정)
- 테스트: `backend/tests/rag/` (구현 예정)
