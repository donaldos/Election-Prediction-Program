from datetime import datetime

from pydantic import BaseModel


class RawArticle(BaseModel):
    """스크레이퍼가 반환하는 원본 기사. 전처리 전 상태."""

    url: str
    source: str

    title: str
    body: str
    published_at: datetime

    candidate: str = ""
    district_id: str = ""

    matched_keywords: list[str] = []


class Article(BaseModel):
    """전처리 완료된 기사. pipeline 내부에서 사용."""

    url: str
    source: str

    title: str
    body: str
    published_at: datetime

    candidate: str
    district_id: str

    matched_keywords: list[str] = []
