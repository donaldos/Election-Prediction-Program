"""RAG 판정 파이프라인 CLI: retrieve → rerank → score.

사용법:
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b
    PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --top-k 10
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --skip-score
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
    )


def _build_reranker(config: dict):
    from rag.reranker import Reranker

    rag_cfg = config.get("rag", {}).get("reranker", {})
    return Reranker(
        min_score=rag_cfg.get("min_score", 0.3),
        deduplicate=rag_cfg.get("deduplicate", True),
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
        print(f"     근거: {s.reasoning}")

    print()
    print(f"  📝 요약: {verdict.summary}")
    print(f"  📰 분석 청크: {verdict.total_chunks_analyzed}건")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Election Radar RAG 판정 파이프라인")
    parser.add_argument(
        "--district", required=True,
        help="선거구 ID (pyeongtaek_b | busan_bukgu_gap)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="검색 수 (config 우선)")
    parser.add_argument("--min-score", type=float, default=None, help="유사도 임계값 (config 우선)")
    parser.add_argument("--skip-score", action="store_true", help="LLM 판정 생략 (검색 결과만)")
    args = parser.parse_args()

    config = _load_config()

    district = _find_district(config, args.district)
    if not district:
        available = [d["id"] for d in config.get("districts", [])]
        logger.error("선거구 '%s' 없음. 사용 가능: %s", args.district, available)
        return

    logger.info("컴포넌트 초기화 중…")
    embedder = _build_embedder(config)
    vector_repo = _build_vector_repo(config)

    if args.top_k:
        config.setdefault("rag", {}).setdefault("retriever", {})["top_k"] = args.top_k
    if args.min_score is not None:
        config.setdefault("rag", {}).setdefault("reranker", {})["min_score"] = args.min_score

    retriever = _build_retriever(config, embedder, vector_repo)
    reranker = _build_reranker(config)

    logger.info("VectorDB 저장 건수: %d", vector_repo.count())

    # retrieve → rerank
    logger.info("선거구 통합 검색 시작 — %s", district["name"])
    all_results = retriever.retrieve_for_district(district)
    query = f"{district['name']} 선거 판세"
    all_results = reranker.rerank(query, all_results)

    _print_chunks(all_results, district)

    if args.skip_score:
        logger.info("--skip-score 지정 — LLM 판정 생략")
        return

    # score
    scorer = _build_scorer(config)
    verdict = scorer.score(all_results, district)

    _print_verdict(verdict)


if __name__ == "__main__":
    main()
