from __future__ import annotations

from datetime import datetime

import pytest

from ingestion.tagger import tag_articles, _count_keyword_hits, _match_article, _build_candidate_map
from models.article import RawArticle


DISTRICTS = [
    {
        "id": "pyeongtaek_b",
        "name": "평택을",
        "candidates": [
            {"name": "김용남", "party": "더불어민주당", "keywords": ["김용남", "평택을", "평택 재보궐"]},
            {"name": "조국", "party": "조국혁신당", "keywords": ["조국", "평택을", "평택 재보궐"]},
            {"name": "유의동", "party": "국민의힘", "keywords": ["유의동", "평택을", "평택 재보궐"]},
        ],
    },
    {
        "id": "busan_bukgu_gap",
        "name": "부산북구갑",
        "candidates": [
            {"name": "하정우", "party": "더불어민주당", "keywords": ["하정우", "부산북구갑", "부산 북구 재보궐"]},
            {"name": "한동훈", "party": "무소속", "keywords": ["한동훈", "부산북구갑", "부산 북구 재보궐"]},
        ],
    },
]


def _make_article(title: str, body: str, **kwargs) -> RawArticle:
    defaults = {
        "url": "https://example.com/test",
        "source": "naver_news",
        "published_at": datetime(2026, 5, 1),
        "candidate": "",
        "district_id": "",
    }
    defaults.update(kwargs)
    return RawArticle(title=title, body=body, **defaults)


# ── _count_keyword_hits ─────────────────────────────

class TestCountKeywordHits:

    def test_single_keyword(self):
        hits = _count_keyword_hits("김용남 후보가 출마했다", ["김용남"])
        assert hits == {"김용남": 1}

    def test_multiple_occurrences(self):
        hits = _count_keyword_hits("김용남 후보 김용남 공약 김용남", ["김용남"])
        assert hits == {"김용남": 3}

    def test_no_match(self):
        hits = _count_keyword_hits("아무 관련 없는 기사", ["김용남", "평택을"])
        assert hits == {}

    def test_multiple_keywords(self):
        hits = _count_keyword_hits("평택을 김용남 후보 평택을", ["김용남", "평택을"])
        assert hits == {"김용남": 1, "평택을": 2}


# ── _build_candidate_map ────────────────────────────

class TestBuildCandidateMap:

    def test_structure(self):
        cmap = _build_candidate_map(DISTRICTS)
        assert "pyeongtaek_b" in cmap
        assert "busan_bukgu_gap" in cmap
        assert len(cmap["pyeongtaek_b"]) == 3
        assert cmap["busan_bukgu_gap"][0]["name"] == "하정우"

    def test_empty_districts(self):
        assert _build_candidate_map([]) == {}


# ── _match_article ──────────────────────────────────

class TestMatchArticle:

    def setup_method(self):
        self.cmap = _build_candidate_map(DISTRICTS)

    def test_single_candidate_match(self):
        text = "한동훈 전 대표가 부산북구갑 출마를 선언했다."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "busan_bukgu_gap"
        assert candidate == "한동훈"
        assert "한동훈" in keywords

    def test_multiple_candidates_same_district(self):
        text = "김용남과 조국이 평택을에서 치열한 접전을 벌이고 있다."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "pyeongtaek_b"
        assert candidate == ""
        assert "김용남" in keywords
        assert "조국" in keywords

    def test_no_match(self):
        text = "오늘 서울 날씨는 맑겠습니다."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == ""
        assert candidate == ""
        assert keywords == []

    def test_cross_district_picks_stronger(self):
        text = "한동훈 한동훈 한동훈 부산북구갑 부산북구갑. 김용남."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "busan_bukgu_gap"

    def test_district_keyword_only(self):
        text = "평택을 재보궐 선거가 다가오고 있다. 평택 재보궐 관심 증가."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "pyeongtaek_b"
        assert candidate == ""


# ── tag_articles ────────────────────────────────────

class TestTagArticles:

    def test_tags_single_candidate(self):
        articles = [_make_article("한동훈 부산북구갑 출마", "한동훈이 출마를 선언했다.")]
        result = tag_articles(articles, DISTRICTS)
        assert result[0].district_id == "busan_bukgu_gap"
        assert result[0].candidate == "한동훈"
        assert len(result[0].matched_keywords) > 0

    def test_tags_comparison_article(self):
        articles = [_make_article(
            "평택을 김용남 vs 조국",
            "김용남 후보와 조국 후보가 치열한 접전을 벌이고 있다.",
        )]
        result = tag_articles(articles, DISTRICTS)
        assert result[0].district_id == "pyeongtaek_b"
        assert result[0].candidate == ""

    def test_skips_already_tagged(self):
        articles = [_make_article(
            "한동훈 부산북구갑",
            "내용",
            candidate="기존후보",
            district_id="existing_district",
        )]
        result = tag_articles(articles, DISTRICTS)
        assert result[0].candidate == "기존후보"
        assert result[0].district_id == "existing_district"

    def test_unmatched_stays_empty(self):
        articles = [_make_article("오늘 날씨", "서울 맑음")]
        result = tag_articles(articles, DISTRICTS)
        assert result[0].candidate == ""
        assert result[0].district_id == ""

    def test_multiple_articles_mixed(self):
        articles = [
            _make_article("한동훈 출마", "한동훈 부산북구갑 선거"),
            _make_article("날씨 기사", "오늘 서울 맑음"),
            _make_article("평택을 판세", "김용남과 유의동 접전 평택을"),
        ]
        result = tag_articles(articles, DISTRICTS)
        assert result[0].district_id == "busan_bukgu_gap"
        assert result[0].candidate == "한동훈"
        assert result[1].district_id == ""
        assert result[2].district_id == "pyeongtaek_b"
        assert result[2].candidate == ""

    def test_empty_articles(self):
        result = tag_articles([], DISTRICTS)
        assert result == []

    def test_empty_districts(self):
        articles = [_make_article("한동훈 출마", "내용")]
        result = tag_articles(articles, [])
        assert result[0].candidate == ""
        assert result[0].district_id == ""

    def test_matched_keywords_populated(self):
        articles = [_make_article("김용남 평택을", "김용남 후보의 공약 발표. 평택 재보궐.")]
        result = tag_articles(articles, DISTRICTS)
        assert "김용남" in result[0].matched_keywords
        assert "평택을" in result[0].matched_keywords or "평택 재보궐" in result[0].matched_keywords
