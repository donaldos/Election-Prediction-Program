from __future__ import annotations

from datetime import datetime

import pytest

SAMPLE_TEXT = """첫 번째 문단입니다. 이 문단은 후보 A에 대한 내용을 담고 있습니다.
평택을 선거구에서 여론조사 결과가 발표되었습니다.

두 번째 문단입니다. 지지율 변동이 감지되었습니다.
후보 B와의 격차가 줄어들고 있다는 분석이 나왔습니다.

세 번째 문단입니다. 전문가들은 막판 변수를 주시하고 있습니다."""

SAMPLE_METADATA = {
    "article_url": "https://example.com/article/1",
    "source": "naver_news",
    "title": "평택을 판세 분석",
    "published_at": datetime(2026, 5, 1, 9, 0),
    "candidate": "후보A",
    "district_id": "pyeongtaek_b",
}


# ── KoreanParagraphChunker ──────────────────────────────────


class TestKoreanParagraphChunker:

    def test_basic(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker(chunk_size=200, overlap=30)
        chunker.load()
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)

        assert len(chunks) >= 1
        assert all(c.chunker_type == "korean_paragraph" for c in chunks)
        assert all(c.candidate == "후보A" for c in chunks)

    def test_empty_text(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker()
        assert chunker.chunk("", SAMPLE_METADATA) == []
        assert chunker.chunk("   ", SAMPLE_METADATA) == []

    def test_whitespace_only(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker()
        assert chunker.chunk("\n\n\n", SAMPLE_METADATA) == []

    def test_single_paragraph(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker(chunk_size=1000)
        chunks = chunker.chunk("짧은 단일 문단입니다.", SAMPLE_METADATA)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_chunk_index_sequential(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_propagation(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker(chunk_size=1000)
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        for c in chunks:
            assert c.article_url == "https://example.com/article/1"
            assert c.source == "naver_news"
            assert c.title == "평택을 판세 분석"
            assert c.district_id == "pyeongtaek_b"

    def test_char_count_matches_text(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        for c in chunks:
            assert c.char_count == len(c.text)

    def test_auto_load_on_chunk(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker()
        assert not chunker.is_loaded
        chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        assert chunker.is_loaded

    def test_overlap_content(self):
        from ingestion.chunker.korean_paragraph import KoreanParagraphChunker

        chunker = KoreanParagraphChunker(chunk_size=80, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        if len(chunks) >= 2:
            tail = chunks[0].text[-20:]
            assert tail in chunks[1].text


# ── RecursiveChunker ───────────────────────────────────────


class TestRecursiveChunker:

    def test_custom_separators(self):
        from ingestion.chunker.recursive import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=150, overlap=20, separators=["\n\n", "\n"])
        chunker.load()
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        assert len(chunks) >= 1

    def test_empty_text(self):
        from ingestion.chunker.recursive import RecursiveChunker

        chunker = RecursiveChunker()
        assert chunker.chunk("", SAMPLE_METADATA) == []

    def test_idempotent_load(self):
        from ingestion.chunker.recursive import RecursiveChunker

        chunker = RecursiveChunker()
        chunker.load()
        chunker.load()
        assert chunker.is_loaded

    def test_short_text_single_chunk(self):
        from ingestion.chunker.recursive import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.chunk("짧은 텍스트", SAMPLE_METADATA)
        assert len(chunks) == 1

    def test_chunk_index_sequential(self):
        from ingestion.chunker.recursive import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_name(self):
        from ingestion.chunker.recursive import RecursiveChunker

        chunker = RecursiveChunker()
        assert chunker.name == "recursive"


# ── SentenceChunker (kss 필요) ────────────────────────────


class TestSentenceChunker:

    def _skip_if_no_kss(self):
        try:
            import kss  # noqa: F401
        except ImportError:
            pytest.skip("kss not installed")

    def test_load(self):
        self._skip_if_no_kss()
        from ingestion.chunker.sentence import SentenceChunker

        chunker = SentenceChunker(sentences_per_chunk=3)
        assert not chunker.is_loaded
        chunker.load()
        assert chunker.is_loaded
        assert chunker._kss is not None

    def test_chunk_index_sequential(self):
        self._skip_if_no_kss()
        from ingestion.chunker.sentence import SentenceChunker

        chunker = SentenceChunker(sentences_per_chunk=2)
        chunker.load()
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text(self):
        self._skip_if_no_kss()
        from ingestion.chunker.sentence import SentenceChunker

        chunker = SentenceChunker()
        assert chunker.chunk("", SAMPLE_METADATA) == []


# ── TokenChunker (tiktoken 필요) ──────────────────────────


class TestTokenChunker:

    def _skip_if_no_tiktoken(self):
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_respects_token_limit(self):
        self._skip_if_no_tiktoken()
        import tiktoken

        from ingestion.chunker.token import TokenChunker

        chunker = TokenChunker(tokens_per_chunk=50, overlap_tokens=10)
        chunker.load()
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)

        enc = tiktoken.get_encoding("cl100k_base")
        for c in chunks:
            assert len(enc.encode(c.text)) <= 60

    def test_empty_text(self):
        self._skip_if_no_tiktoken()
        from ingestion.chunker.token import TokenChunker

        chunker = TokenChunker()
        assert chunker.chunk("", SAMPLE_METADATA) == []

    def test_load(self):
        self._skip_if_no_tiktoken()
        from ingestion.chunker.token import TokenChunker

        chunker = TokenChunker()
        assert not chunker.is_loaded
        chunker.load()
        assert chunker.is_loaded


# ── SemanticChunker (모델 다운로드 필요 — 기본 skip) ──────


class TestSemanticChunker:

    @pytest.mark.skip(reason="모델 다운로드 필요 — 로컬에서만 실행")
    def test_load_sets_model(self):
        from ingestion.chunker.semantic import SemanticChunker

        chunker = SemanticChunker(model_name="BAAI/bge-m3")
        assert not chunker.is_loaded
        chunker.load()
        assert chunker.is_loaded
        assert chunker._model is not None

    @pytest.mark.skip(reason="모델 다운로드 필요 — 로컬에서만 실행")
    def test_splits_on_topic_change(self):
        from ingestion.chunker.semantic import SemanticChunker

        chunker = SemanticChunker(breakpoint_threshold=0.5)
        chunker.load()
        chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
        assert len(chunks) >= 1

    def test_empty_text_no_model(self):
        from ingestion.chunker.semantic import SemanticChunker

        chunker = SemanticChunker()
        chunker._loaded = True
        assert chunker.chunk("", SAMPLE_METADATA) == []


# ── ChunkerRegistry ──────────────────────────────────────


class TestChunkerRegistry:

    def test_all_registered(self):
        import ingestion.chunker  # noqa: F401
        from ingestion.chunker.base import ChunkerRegistry

        names = ChunkerRegistry.registered_names
        assert "korean_paragraph" in names
        assert "sentence" in names
        assert "token" in names
        assert "semantic" in names
        assert "recursive" in names

    def test_create_by_name(self):
        import ingestion.chunker  # noqa: F401
        from ingestion.chunker.base import ChunkerRegistry

        chunker = ChunkerRegistry.create("korean_paragraph", chunk_size=300, overlap=40)
        assert chunker.name == "korean_paragraph"

    def test_create_unknown_raises(self):
        from ingestion.chunker.base import ChunkerRegistry

        with pytest.raises(ValueError, match="미등록"):
            ChunkerRegistry.create("nonexistent_chunker")
