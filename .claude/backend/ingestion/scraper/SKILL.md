---
name: scraper
description: >
  Election Radar 프로젝트의 뉴스 스크레이퍼 컴포넌트를 구현하거나 수정할 때 사용.
  NaverNewsScraper, PoliticalNewsScraper 추가·변경, AbstractScraper 인터페이스 수정,
  ScraperRegistry 등록, 크롤링 설정(config.yaml) 변경, robots.txt 준수 정책 확인,
  스크레이퍼 테스트 작성 시 반드시 이 파일을 먼저 읽으세요.
---

# Scraper 컴포넌트 가이드

## 역할

뉴스 원문을 수집하여 `RawArticle` 모델로 반환합니다.
스크레이퍼는 **수집만** 담당합니다. 전처리·청킹·임베딩은 pipeline.py가 담당합니다.

```
[config.yaml 키워드 + lookback_days]
       ↓
AbstractScraper.scrape(keywords, date_from?, date_to?) → list[RawArticle]
       ↓
ScrapedUrlStore에 URL 저장 (중복 방지)
       ↓
ingestion/pipeline.py (청킹·임베딩으로 전달)
```

---

## 파일 구조

```
ingestion/scraper/
├── base.py          ← AbstractScraper ABC + ScraperRegistry (이 파일 먼저 읽기)
├── naver.py         ← NaverNewsScraper (네이버 검색 HTML 파싱)
├── political.py     ← PoliticalNewsScraper (오마이뉴스·프레시안 등 RSS 파싱)
├── url_store.py     ← ScrapedUrlStore (수집 URL JSONL 영속 저장)
├── run.py           ← 수동 실행 스크립트 (CLI)
└── __init__.py      ← 구현체 자동 등록을 위한 import
```

> **`__init__.py` 주의**: 새 스크레이퍼를 추가하면 반드시 `__init__.py`에 import를 추가해야
> `@ScraperRegistry.register` 데코레이터가 실행되어 등록됩니다.

---

## AbstractScraper 인터페이스

```python
# ingestion/scraper/base.py

class AbstractScraper(ABC):

    @abstractmethod
    def scrape(
        self,
        keywords: list[str],
        date_from: date | None = None,
        date_to: date | None = None,
        max_articles: int = 50,
    ) -> list[RawArticle]:
        """
        date_from/date_to가 None이면 lookback_days 기준 자동 설정.
        실패 시 빈 리스트 반환 (예외 raise 금지).
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @property
    def request_delay_sec(self) -> float:
        return 1.5

    @property
    def lookback_days(self) -> int:
        return 2  # 기본값: 오늘 기준 2일 전부터 검색

    def resolve_date_range(
        self, date_from: date | None, date_to: date | None
    ) -> tuple[date, date]:
        """None이면 오늘 기준 lookback_days일 전~오늘로 설정."""
        ...
```

---

## URL 영속 저장 (ScrapedUrlStore)

수집된 기사 URL을 `data/scraped_urls.jsonl`에 JSONL로 저장하여 중복 수집을 방지합니다.

```python
# ingestion/scraper/url_store.py

store = ScrapedUrlStore()                    # 기본: data/scraped_urls.jsonl
store.contains("https://...")                # 이미 수집된 URL인지 확인
store.add(url, source, title)                # 단건 저장
store.add_batch([{"url":..., ...}])          # 배치 저장, 신규 건수 반환
```

**JSONL 레코드 형식:**
```json
{"url": "https://...", "source": "naver_news", "title": "기사 제목", "scraped_at": "2026-04-28T15:17:38"}
```

---

## 수동 실행 (run.py)

```bash
# 전체 스크레이퍼 (네이버 + 정치매체, lookback_days 기준)
PYTHONPATH=. python -m ingestion.scraper.run

# 네이버만
PYTHONPATH=. python -m ingestion.scraper.run --scraper naver

# 정치 매체만
PYTHONPATH=. python -m ingestion.scraper.run --scraper political

# 검색 기간 변경 (5일 전부터)
PYTHONPATH=. python -m ingestion.scraper.run --days 5
```

**결과 파일:**
- `data/scraped_urls.jsonl` — URL 기록 (영속, 중복 방지)
- `data/articles_YYYY-MM-DD_HHMMSS.jsonl` — 일자별 수집 기사 전문

---

## NaverNewsScraper 구현 노트

네이버 뉴스는 공식 API가 없으므로 검색 결과 페이지를 파싱합니다.

### 사용 라이브러리

```python
# lazy import — __init__에서 하지 말 것
import httpx
from bs4 import BeautifulSoup
```

### 검색 URL 패턴

```
https://search.naver.com/search.naver
  ?where=news
  &query={keyword}
  &ds={date_from}    # YYYY.MM.DD 형식
  &de={date_to}
  &sort=1            # 최신순
  &start={offset}    # 페이지네이션 (10씩 증가)
```

### 파싱 대상 셀렉터 (2026년 4월 기준, 변경 가능)

네이버는 SDS 디자인 시스템으로 전환하여 HTML 구조가 변경되었습니다.
셀렉터는 `naver.py` 상단에 상수로 관리합니다.

```python
# 기사 컨테이너 (각 기사를 감싸는 div)
SEL_ARTICLE_CONTAINER = 'div[class*="qhLRRX"]'

# 기사 제목 (a 태그, href에 원문 URL 포함)
SEL_TITLE = 'a[data-heatmap-target=".tit"]'

# 본문 요약 (a 태그)
SEL_SUMMARY = 'a[data-heatmap-target=".body"]'

# 언론사명 (a > span)
SEL_PRESS = 'a[data-heatmap-target=".prof"] span'

# 날짜 ("2026.05.01.", "5분 전", "2시간 전" 등)
SEL_DATE = 'span.sds-comps-text-ellipsis-1'
```

**날짜 파싱**: `"N분 전"`, `"N시간 전"`, `"N일 전"`, `"YYYY.MM.DD."` 형식 모두 지원.

> **셀렉터 변경 대응**: 네이버는 주기적으로 HTML 구조를 변경합니다.
> 수집 결과가 0건이면 실제 검색 페이지의 HTML을 확인하고 셀렉터를 업데이트하세요.
> `data-heatmap-target` 속성 기반 셀렉터가 클래스명 기반보다 안정적입니다.

### 헤더 설정 (필수)

```python
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://www.naver.com",
}
```

### robots.txt 준수 규칙

- 요청 간 **최소 1.5초** 딜레이 (`request_delay_sec` 기본값)
- 동시 요청 금지 — 반드시 순차 처리
- 하루 총 수집량은 `max_articles_per_run` × 크롤링 횟수(3회)로 제한
- 404 / 429 응답 시 즉시 중단하고 logger.warning 기록

---

## PoliticalNewsScraper 구현 노트

여러 정치 전문 매체를 하나의 스크레이퍼로 처리합니다.
`config.yaml`의 `urls` 목록을 순회합니다.

### 지원 대상 매체 (우선순위 순)

| 매체 | RSS 지원 | 비고 |
|------|---------|------|
| 오마이뉴스 | ✅ | RSS 파싱 권장 |
| 프레시안 | ✅ | RSS 파싱 권장 |
| 미디어오늘 | ❌ | HTML 파싱 필요 |

### RSS 우선 전략

RSS가 있는 매체는 HTML 파싱 대신 RSS를 파싱합니다.
파싱 라이브러리: `feedparser` (lazy import)

---

## 중복 제거 처리

중복 제거는 두 단계로 이루어집니다:

1. **ScrapedUrlStore** (실행 간 중복 방지): 이전 실행에서 이미 수집한 URL을 스킵
2. **scrape() 내부 seen_urls** (실행 내 중복 방지): 같은 실행에서 여러 키워드에 걸린 동일 기사 제거

---

## 새 스크레이퍼 구현 체크리스트

### 1단계: 구현 파일 생성

```python
# ingestion/scraper/my_source.py

@ScraperRegistry.register("my_source")
class MySourceScraper(AbstractScraper):
    def __init__(self, base_url: str, ..., url_store_path=None) -> None:
        self._url_store = ScrapedUrlStore(url_store_path)
        ...
```

### 2단계: `__init__.py`에 등록

```python
from ingestion.scraper import my_source   # ← 추가
```

### 3단계: config.yaml 추가

```yaml
scrapers:
  my_source:
    type: my_source
    params:
      base_url: "https://my-news-source.com"
      max_articles_per_run: 100
      lookback_days: 2
```

### 4단계: 테스트 작성 (`tests/ingestion/test_scraper.py` 참조)

---

## 테스트 작성 가이드

실제 HTTP 요청은 테스트에서 **절대 사용하지 않습니다**.
`unittest.mock.patch("httpx.get")`으로 mock합니다.

```python
def test_naver_scraper_returns_articles(tmp_path):
    scraper = NaverNewsScraper(
        max_articles_per_run=10,
        request_delay_sec=0,              # 테스트에서는 딜레이 0
        url_store_path=tmp_path / "u.jsonl",  # 임시 경로
    )
    with patch("httpx.get", side_effect=_mock_httpx_get):
        results = scraper.scrape(keywords=["평택을"], date_from=..., date_to=...)
    assert len(results) > 0
```

**mock HTML은 네이버 SDS 구조를 따라야 합니다:**
```html
<div class="qhLRRX desktop_mode">
  <a data-heatmap-target=".prof" href="..."><span class="...">언론사</span></a>
  <a data-heatmap-target=".tit" href="https://...">기사 제목</a>
  <a data-heatmap-target=".body" href="...">기사 요약</a>
  <span class="sds-comps-text sds-comps-text-ellipsis sds-comps-text-ellipsis-1">2026.05.01.</span>
</div>
```

---

## 로깅 규칙

| 레벨 | 내용 |
|------|------|
| `INFO` | 초기화 설정, 수집 시작/완료 통계, 키워드별 결과, limit 도달, URL 저장 건수 |
| `WARNING` | HTTP 오류(404/429), 수집 중 예외 |
| `DEBUG` | 개별 HTTP 요청, 페이지 파싱 결과, 날짜 파싱 실패, RSS 엔트리 수 |

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| 모듈 최상단에서 `import httpx` | 메서드 내부에서 lazy import |
| HTTP 오류 시 `raise` | `logger.warning` 후 `return []` |
| `__init__.py`에 import 누락 | 새 파일 추가 시 반드시 `__init__.py` 업데이트 |
| 요청 딜레이 없음 | `time.sleep(self.request_delay_sec)` 키워드 루프마다 호출 |
| 셀렉터 하드코딩 후 변경 미반영 | 셀렉터를 모듈 상수로 분리하여 한 곳에서 관리 |
| 테스트에서 실제 HTTP 요청 | 항상 `unittest.mock.patch("httpx.get")`로 mock |
| URL 저장소 미연동 | `ScrapedUrlStore`로 중복 수집 방지 |

---

## 관련 파일 참조

- 전체 아키텍처: `CLAUDE.md`
- 범용 Registry: `backend/ingestion/base_registry.py`
- 도메인 모델: `backend/models/article.py`
- URL 저장소: `backend/ingestion/scraper/url_store.py`
- 수동 실행: `backend/ingestion/scraper/run.py`
- config.yaml: `backend/config/config.yaml`
- 테스트: `backend/tests/ingestion/test_scraper.py`
