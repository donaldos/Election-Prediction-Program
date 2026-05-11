"""RAG 판정 파이프라인 CLI: retrieve → rerank → score.

사용법:
    # 판정 모드 (기존)
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b
    PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap

    # 단발 질의 모드
    PYTHONPATH=. python -m rag.pipeline --query "조국의 평택을 지지율 변화는?"
    PYTHONPATH=. python -m rag.pipeline --query "한동훈과 박민식 비교" --top-k 10
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.pipeline")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_district(config: dict, district_id: str) -> dict | None:
    for d in config.get("districts", []):
        if d["id"] == district_id:
            return d
    return None


def _build_embedder(config: dict):
    import ingestion.embedder  # noqa: F401
    from ingestion.embedder.base import EmbedderRegistry

    cfg = config.get("embedder", {})
    embedder = EmbedderRegistry.create(cfg.get("type", "openai"), **cfg.get("params", {}))
    embedder.load()
    return embedder


def _build_vector_repo(config: dict):
    import vectordb  # noqa: F401
    from vectordb.base import VectorRepositoryRegistry

    cfg = config.get("vectordb", {})
    repo = VectorRepositoryRegistry.create(
        cfg.get("type", "chroma"),
        collection=cfg.get("collection", "election_chunks"),
        **cfg.get("params", {}),
    )
    repo.load()
    return repo


def _build_retriever(config: dict, embedder, vector_repo):
    from rag.retriever import Retriever

    rag_cfg = config.get("rag", {}).get("retriever", {})
    return Retriever(
        embedder=embedder,
        vector_repo=vector_repo,
        top_k=rag_cfg.get("top_k", 20),
        top_k_common=rag_cfg.get("top_k_common"),
        top_k_candidate=rag_cfg.get("top_k_candidate"),
        lookback_days=rag_cfg.get("lookback_days"),
    )


def _build_reranker(config: dict):
    from rag.reranker import Reranker

    rag_cfg = config.get("rag", {}).get("reranker", {})
    ce_cfg = rag_cfg.get("cross_encoder", {})
    ce_model = ce_cfg.get("model") if ce_cfg.get("enabled") else None
    return Reranker(
        min_score=rag_cfg.get("min_score", 0.3),
        deduplicate=rag_cfg.get("deduplicate", True),
        cross_encoder_model=ce_model,
        cross_encoder_top_n=ce_cfg.get("top_n"),
    )


def _build_scorer(config: dict):
    import rag.openai_scorer  # noqa: F401
    import rag.anthropic_scorer  # noqa: F401
    from rag.scorer import ScorerRegistry

    rag_cfg = config.get("rag", {}).get("scorer", {})
    provider = rag_cfg.get("provider", "openai")
    params = {
        k: v for k, v in rag_cfg.items()
        if k not in ("provider",)
    }
    return ScorerRegistry.create(provider, **params)


QA_SYSTEM_PROMPT = """\
당신은 한국 선거 뉴스 분석 전문가입니다.
제공된 뉴스 기사 청크를 근거로 사용자의 질문에 답변합니다.

규칙:
- 제공된 뉴스 청크에 근거하여 답변하세요. 근거가 없는 내용은 "제공된 기사에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
- 답변은 한국어로 작성하세요.
- 출처(기사 제목, 매체)를 명시하세요.
- 간결하고 핵심적으로 답변하세요."""


def _build_qa_prompt(query: str, chunks) -> str:
    lines = [f"## 사용자 질문\n{query}\n"]
    lines.append(f"## 참고 뉴스 청크 ({len(chunks)}건)\n")
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {chunk.title} ({chunk.source}, {chunk.published_at:%Y-%m-%d})"
        )
        text_preview = chunk.text[:500].replace("\n", " ")
        lines.append(f"    {text_preview}")
        lines.append("")
    lines.append("위 뉴스를 근거로 사용자의 질문에 답변하세요.")
    return "\n".join(lines)


def _print_qa_answer(query: str, answer: str, chunk_count: int) -> None:
    print()
    print("=" * 60)
    print(f"❓ 질문: {query}")
    print("=" * 60)
    print()
    print(answer)
    print()
    print(f"  📰 참고 청크: {chunk_count}건")
    print("=" * 60)
    print()


def _print_chunks(results, district: dict) -> None:
    print()
    print(f"  검색 결과: {len(results)}건")
    for i, r in enumerate(results[:5], 1):
        text_preview = r.text[:100].replace("\n", " ")
        if len(r.text) > 100:
            text_preview += "…"
        print(f"    [{i}] score={r.score:.4f} | {r.title[:40]}")
        print(f"        {text_preview}")
    if len(results) > 5:
        print(f"    ... 외 {len(results) - 5}건")


_REASONING_LABELS = [
    ("support_rate",    "📊 지지율"),
    ("pledge_reaction", "📋 공약 반응"),
    ("strengths",       "💪 강점"),
    ("weaknesses",      "⚠️ 약점"),
    ("issues",          "🔥 이슈"),
    ("support_trend",   "📈 지지율 추이"),
    ("public_opinion",  "🗳️ 출마 여론"),
    ("strategy",        "🎯 선거 전략"),
    ("forecast",        "🔮 예측"),
]


def _print_verdict(verdict) -> None:
    print()
    print("=" * 60)
    print(f"📊 판정 결과: {verdict.district_name} ({verdict.date:%Y-%m-%d %H:%M})")
    print("=" * 60)

    for s in sorted(verdict.candidates, key=lambda c: c.win_probability, reverse=True):
        bar_len = int(s.win_probability * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        verdict_emoji = {"우세": "🔴", "균형": "🟡", "열세": "🔵"}.get(s.verdict, "⚪")

        print(f"\n  {verdict_emoji} {s.candidate} ({s.party})")
        print(f"     판정: {s.verdict}  |  승률: {s.win_probability:.1%}")
        print(f"     {bar}")

        if isinstance(s.reasoning, str):
            print(f"     근거: {s.reasoning}")
        else:
            for field, label in _REASONING_LABELS:
                value = getattr(s.reasoning, field, "")
                if value:
                    print(f"     {label}: {value}")

    print()
    print(f"  📝 요약: {verdict.summary}")
    print(f"  📰 분석 청크: {verdict.total_chunks_analyzed}건")
    print("=" * 60)
    print()


def _run_query_mode(args, config: dict) -> None:
    """단발 질의 모드: 사용자 질문 → VectorDB 검색 → LLM 답변."""
    import time

    logger.info("컴포넌트 초기화 중…")
    embedder = _build_embedder(config)
    vector_repo = _build_vector_repo(config)

    if args.top_k:
        config.setdefault("rag", {}).setdefault("retriever", {})["top_k"] = args.top_k
    if args.min_score is not None:
        config.setdefault("rag", {}).setdefault("reranker", {})["min_score"] = args.min_score
    if args.lookback_days is not None:
        config.setdefault("rag", {}).setdefault("retriever", {})["lookback_days"] = args.lookback_days

    retriever = _build_retriever(config, embedder, vector_repo)
    reranker = _build_reranker(config)

    logger.info("VectorDB 저장 건수: %d", vector_repo.count())

    query = args.query
    logger.info("단발 질의 검색 — '%s'", query)

    from rag.retriever import Retriever
    query_vector = embedder.embed_query(query)
    raw_results = vector_repo.search(
        query_vector=query_vector,
        top_k=config.get("rag", {}).get("retriever", {}).get("top_k", 20),
        filters=None,
    )

    from models.score import SearchResult
    results = []
    for r in raw_results:
        try:
            results.append(SearchResult(
                id=r["id"],
                score=r["score"],
                text=r.get("text", ""),
                article_url=r.get("article_url", ""),
                source=r.get("source", ""),
                title=r.get("title", ""),
                published_at=r.get("published_at"),
                candidate=r.get("candidate", ""),
                district_id=r.get("district_id", ""),
            ))
        except Exception:
            continue

    results = reranker.rerank(query, results)

    if not results:
        print("\n  검색 결과가 없습니다. 수집 파이프라인을 먼저 실행하세요.\n")
        return

    _print_chunks(results, {"name": "전체"})

    scorer = _build_scorer(config)
    user_prompt = _build_qa_prompt(query, results)

    logger.info("LLM 질의 응답 요청 — 청크 %d건", len(results))
    t0 = time.monotonic()
    answer = scorer._call_llm(QA_SYSTEM_PROMPT, user_prompt, json_mode=False)
    elapsed = time.monotonic() - t0
    logger.info("LLM 응답 수신 — %.1f초", elapsed)

    _print_qa_answer(query, answer, len(results))


def _run_verdict_mode(args, config: dict) -> None:
    """판정 모드: 선거구별 판세 분석."""
    district = _find_district(config, args.district)
    if not district:
        available = [d["id"] for d in config.get("districts", [])]
        logger.error("선거구 '%s' 없음. 사용 가능: %s", args.district, available)
        return

    logger.info("컴포넌트 초기화 중…")
    embedder = _build_embedder(config)
    vector_repo = _build_vector_repo(config)

    purge_days = args.purge_days or config.get("rag", {}).get("purge_days")
    if purge_days:
        deleted = vector_repo.delete_older_than(purge_days)
        logger.info("만료 정리 완료 — %d개 삭제 (%d일 이전)", deleted, purge_days)

    if args.top_k:
        config.setdefault("rag", {}).setdefault("retriever", {})["top_k"] = args.top_k
    if args.min_score is not None:
        config.setdefault("rag", {}).setdefault("reranker", {})["min_score"] = args.min_score
    if args.lookback_days is not None:
        config.setdefault("rag", {}).setdefault("retriever", {})["lookback_days"] = args.lookback_days

    retriever = _build_retriever(config, embedder, vector_repo)
    reranker = _build_reranker(config)

    logger.info("VectorDB 저장 건수: %d", vector_repo.count())

    logger.info("선거구 그룹별 검색 시작 — %s", district["name"])
    grouped = retriever.retrieve_for_district_grouped(district)
    grouped = reranker.rerank_grouped(grouped, district_name=district["name"])

    from rag.scorer import flatten_grouped_chunks
    flat_chunks = flatten_grouped_chunks(grouped)
    _print_chunks(flat_chunks, district)

    if args.skip_score:
        logger.info("--skip-score 지정 — LLM 판정 생략")
        return

    scorer = _build_scorer(config)
    verdict = scorer.score(flat_chunks, district, grouped_chunks=grouped)

    from rag.verdict_store import VerdictStore
    store = VerdictStore()
    store.save(verdict)

    _print_verdict(verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Election Radar RAG 판정 파이프라인")
    parser.add_argument("--district", default=None, help="선거구 ID (pyeongtaek_b | busan_bukgu_gap)")
    parser.add_argument("--query", default=None, help="단발 질의 모드 — 자유 질문")
    parser.add_argument("--top-k", type=int, default=None, help="검색 수 (config 우선)")
    parser.add_argument("--min-score", type=float, default=None, help="유사도 임계값 (config 우선)")
    parser.add_argument("--lookback-days", type=int, default=None, help="최근 N일 기사만 검색 (config 우선)")
    parser.add_argument("--purge-days", type=int, default=None, help="N일 이전 벡터 삭제")
    parser.add_argument("--skip-score", action="store_true", help="LLM 판정 생략 (검색 결과만)")
    args = parser.parse_args()

    if not args.query and not args.district:
        parser.error("--district 또는 --query 중 하나를 지정하세요.")

    config = _load_config()

    if args.query:
        _run_query_mode(args, config)
    else:
        _run_verdict_mode(args, config)


if __name__ == "__main__":
    main()
