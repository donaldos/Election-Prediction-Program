from __future__ import annotations

from datetime import datetime

import pytest

from ingestion.tagger import (
    tag_articles, _count_keyword_hits, _match_article,
    _build_candidate_map, _has_election_context,
)
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


# ── _has_election_context ──────────────────────────

class TestHasElectionContext:

    def test_election_word_nearby(self):
        text = "조국 후보가 출마를 선언했다"
        assert _has_election_context(text, 0, 2) is True

    def test_no_election_word(self):
        text = "가슴속에 늘 조국을 품고 살아온 핏줄이다"
        idx = text.index("조국")
        assert _has_election_context(text, idx, idx + 2) is False

    def test_party_name_as_context(self):
        text = "조국혁신당에서 활동하는 조국 대표"
        assert _has_election_context(text, 0, 2) is True

    def test_context_window_boundary(self):
        text = ("가" * 60) + "조국" + ("나" * 60) + "후보"
        idx = text.index("조국")
        assert _has_election_context(text, idx, idx + 2) is False


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

    def test_require_context_filters_non_election(self):
        text = "가슴속에 늘 조국을 품고 살아온 핏줄"
        hits = _count_keyword_hits(text, ["조국"], require_context=True)
        assert hits == {}

    def test_require_context_keeps_election(self):
        text = "조국 후보가 평택을 재보궐 선거에 출마했다"
        hits = _count_keyword_hits(text, ["조국"], require_context=True)
        assert hits == {"조국": 1}

    def test_require_context_partial_match(self):
        padding = "이것은 관련 없는 내용입니다. " * 5
        text = f"조국을 사랑합니다. {padding}조국 후보가 출마했습니다."
        hits = _count_keyword_hits(text, ["조국"], require_context=True)
        assert hits == {"조국": 1}


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
        text = "김용남 후보와 조국 후보가 평택을에서 치열한 접전을 벌이고 있다."
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
        text = "한동훈 후보 한동훈 후보 한동훈 후보 부산북구갑 선거. 김용남 후보."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "busan_bukgu_gap"

    def test_district_keyword_only(self):
        text = "평택을 재보궐 선거가 다가오고 있다. 평택 재보궐 관심 증가."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "pyeongtaek_b"
        assert candidate == ""

    def test_no_match_without_election_context(self):
        text = "고려인은 가슴속에 늘 조국을 품고 살아온 우리의 소중한 핏줄이다."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == ""
        assert candidate == ""

    def test_match_with_election_context(self):
        text = "조국 대표가 평택을 재보궐 선거에 출마를 선언했다."
        district_id, candidate, keywords = _match_article(text, self.cmap)
        assert district_id == "pyeongtaek_b"
        assert candidate == "조국"


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
            "평택을 김용남 vs 조국 선거",
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
            _make_article("한동훈 출마 선언", "한동훈 후보가 부산북구갑 선거에 출마"),
            _make_article("날씨 기사", "오늘 서울 맑음"),
            _make_article("평택을 판세", "김용남 후보와 유의동 후보 접전 평택을 재보궐"),
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
