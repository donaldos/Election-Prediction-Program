"""RAG 판정 파이프라인 CLI: retrieve → rerank → score.

사용법:
    # 판세 판정 (기본)
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b
    PYTHONPATH=. python -m rag.pipeline --district busan_bukgu_gap

    # 문제점 진단
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --analysis diagnosis

    # 대응방안/전략 도출
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --analysis strategy

    # 후보 간 비교 분석
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --analysis comparison --candidates "김용남,유의동"

    # 특정 후보만 진단
    PYTHONPATH=. python -m rag.pipeline --district pyeongtaek_b --analysis diagnosis --candidates "김용남"

    # 단발 질의 모드
    PYTHONPATH=. python -m rag.pipeline --query "조국의 평택을 지지율 변화는?"
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
제공된 뉴스 기사 청크와 여론조사 데이터를 근거로 사용자의 질문에 답변합니다.

규칙:
- 제공된 뉴스 청크 및 여론조사 데이터에 근거하여 답변하세요. 근거가 없는 내용은 "제공된 자료에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
- 여론조사 데이터가 포함된 경우, 시계열 변동 추이를 중심으로 분석하세요. 조사일자별 지지율 변화, 추세(상승/하락/정체), 변곡점과 원인을 설명하세요.
- 조사 방법론(무선전화면접 vs ARS)이 다른 조사 간 단순 수치 비교를 지양하고, 동일 조사기관의 시계열 변화를 우선 비교하세요.
- 오차범위 내 변동은 "추세 변화"로 해석하지 마세요.
- 답변은 한국어로 작성하세요.
- 출처(기사 제목, 매체, 조사기관)를 명시하세요.
- 간결하고 핵심적으로 답변하세요."""


_POLL_KEYWORDS = {"여론조사", "지지율", "추이", "동향", "변화", "변동", "조사", "서베이", "갤럽", "리서치"}


def _detect_poll_districts(query: str, config: dict) -> list[dict]:
    """쿼리에서 여론조사 관련 키워드와 선거구를 감지한다."""
    if not any(kw in query for kw in _POLL_KEYWORDS):
        return []

    matched: list[dict] = []
    for district in config.get("districts", []):
        name = district["name"]
        all_keywords = [name, district["id"]]
        for cand in district.get("candidates", []):
            all_keywords.append(cand["name"])
            all_keywords.extend(cand.get("keywords", []))
        if any(kw in query for kw in all_keywords) or any(
            query_token in kw for query_token in query.split() for kw in all_keywords
        ):
            matched.append(district)

    if not matched:
        matched = config.get("districts", [])
    return matched


def _format_poll_trend_table(
    poll_entries: list,
    poll_metas: list,
    district: dict,
) -> str:
    """여론조사 데이터를 시계열 표 형태로 포맷한다."""
    from collections import defaultdict

    if not poll_entries:
        return ""

    lines = [f"## 여론조사 시계열 데이터 — {district['name']}\n"]

    survey_groups: dict[tuple[str, str], list] = defaultdict(list)
    for entry in sorted(poll_entries, key=lambda e: (e.survey_date, e.pollster)):
        key = (str(entry.survey_date), entry.pollster)
        survey_groups[key].append(entry)

    meta_map: dict[tuple[str, str], object] = {}
    for m in poll_metas:
        meta_map[(str(m.survey_date), m.pollster)] = m

    candidates_seen: list[str] = []
    for entries in survey_groups.values():
        for e in entries:
            if e.candidate not in candidates_seen:
                candidates_seen.append(e.candidate)

    header = "| 조사일 | 조사기관 | 방법 |"
    separator = "|--------|----------|------|"
    for cand in candidates_seen:
        header += f" {cand} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    for (date_str, pollster), entries in survey_groups.items():
        meta = meta_map.get((date_str, pollster))
        method = meta.method if meta and meta.method else "-"
        support_map = {e.candidate: e.support for e in entries}
        row = f"| {date_str} | {pollster} | {method} |"
        for cand in candidates_seen:
            val = support_map.get(cand)
            row += f" {val}% |" if val is not None else " - |"
        lines.append(row)

    lines.append("")

    if poll_metas:
        latest_meta = max(poll_metas, key=lambda m: m.survey_date)
        if latest_meta.sample_size:
            lines.append(
                f"※ 최근 조사 기준 — 표본: {latest_meta.sample_size}명, "
                f"오차범위: ±{latest_meta.margin_of_error}%p"
            )
    lines.append("")
    return "\n".join(lines)


def _build_qa_prompt(
    query: str,
    chunks,
    poll_sections: list[str] | None = None,
) -> str:
    lines = [f"## 사용자 질문\n{query}\n"]

    if poll_sections:
        for section in poll_sections:
            lines.append(section)

    if chunks:
        lines.append(f"## 참고 뉴스 청크 ({len(chunks)}건)\n")
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] {chunk.title} ({chunk.source}, {chunk.published_at:%Y-%m-%d})"
            )
            text_preview = chunk.text[:500].replace("\n", " ")
            lines.append(f"    {text_preview}")
            lines.append("")

    lines.append("위 자료를 근거로 사용자의 질문에 답변하세요.")
    if poll_sections:
        lines.append("여론조사 데이터가 있으면 시계열 변동 추이를 중심으로 분석하세요.")
    return "\n".join(lines)


def _print_poll_chart(
    poll_entries: list,
    poll_metas: list,
    district: dict,
) -> None:
    """여론조사 시계열 ASCII 차트를 출력한다."""
    from collections import defaultdict

    if not poll_entries:
        return

    survey_groups: dict[tuple[str, str], dict[str, float]] = {}
    survey_order: list[tuple[str, str]] = []
    for entry in sorted(poll_entries, key=lambda e: (e.survey_date, e.pollster)):
        key = (str(entry.survey_date), entry.pollster)
        if key not in survey_groups:
            survey_groups[key] = {}
            survey_order.append(key)
        survey_groups[key][entry.candidate] = entry.support

    candidates: list[str] = []
    for group in survey_groups.values():
        for c in group:
            if c not in candidates:
                candidates.append(c)

    markers = ["●", "■", "▲", "◆", "★", "○", "□", "△"]
    all_values = [v for g in survey_groups.values() for v in g.values()]
    y_min = max(0, (int(min(all_values)) // 5) * 5)
    y_max = min(100, (int(max(all_values)) + 5) // 5 * 5)
    chart_height = max(8, min(14, (y_max - y_min) // 2))
    n_surveys = len(survey_order)
    col_w = max(8, 60 // n_surveys)

    print()
    print("=" * 60)
    print(f"📊 {district['name']} 여론조사 추이")
    print("=" * 60)

    legend = "  "
    for i, cand in enumerate(candidates):
        legend += f"{markers[i % len(markers)]} {cand}  "
    print(legend)
    print()

    grid: dict[tuple[int, int], list[str]] = defaultdict(list)
    for col_idx, key in enumerate(survey_order):
        group = survey_groups[key]
        for cand_idx, cand in enumerate(candidates):
            if cand not in group:
                continue
            val = group[cand]
            row = round((val - y_min) / (y_max - y_min) * chart_height)
            row = max(0, min(chart_height, row))
            grid_row = chart_height - row
            grid[(grid_row, col_idx)].append(markers[cand_idx % len(markers)])

    for row in range(chart_height + 1):
        y_val = y_max - row * (y_max - y_min) / chart_height
        label = f"{y_val:4.0f}% │"
        line = f"  {label}"
        for col in range(n_surveys):
            cell_markers = grid.get((row, col), [])
            if cell_markers:
                cell = "".join(cell_markers)
            else:
                cell = " "
            pad_left = col_w // 2
            pad_right = col_w - pad_left - len(cell)
            line += " " * pad_left + cell + " " * max(0, pad_right)
        print(line)

    axis_line = "  " + " " * 7 + "└" + "─" * (n_surveys * col_w)
    print(axis_line)

    date_line = "  " + " " * 8
    for date_str, _ in survey_order:
        date_line += date_str[5:].center(col_w)
    print(date_line)

    pollster_line = "  " + " " * 8
    for _, pollster in survey_order:
        short = pollster if len(pollster) <= col_w - 1 else pollster[: col_w - 2] + "…"
        pollster_line += short.center(col_w)
    print(pollster_line)

    print()
    for i, cand in enumerate(candidates):
        values = []
        for key in survey_order:
            v = survey_groups[key].get(cand)
            if v is not None:
                values.append(v)
        if len(values) >= 2:
            diff = values[-1] - values[0]
            arrow = "▲" if diff > 1 else ("▼" if diff < -1 else "─")
            trend = f"{arrow} {diff:+.1f}%p"
        else:
            trend = ""
        m = markers[i % len(markers)]
        latest = values[-1] if values else 0
        print(f"  {m} {cand}: 최신 {latest:.1f}%  {trend}")

    print("=" * 60)
    print()


def _print_qa_answer(query: str, answer: str, chunks) -> None:
    print()
    print("=" * 60)
    print(f"❓ 질문: {query}")
    print("=" * 60)
    print()
    print(answer)
    print()
    print(f"  📰 참고 청크: {len(chunks)}건")
    print("-" * 60)
    for i, c in enumerate(chunks, 1):
        date_str = c.published_at.strftime("%Y-%m-%d") if c.published_at else ""
        print(f"  [{i}] {c.title}")
        print(f"      출처: {c.source} | {date_str} | score: {c.score:.4f}")
        if c.article_url:
            print(f"      링크: {c.article_url}")
        text_preview = c.text[:150].replace("\n", " ")
        if len(c.text) > 150:
            text_preview += "…"
        print(f"      내용: {text_preview}")
        print()
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

    poll_sections: list[str] = []
    poll_data_for_chart: list[tuple[list, list, dict]] = []
    poll_districts = _detect_poll_districts(query, config)
    if poll_districts:
        from rag.poll_store import create_poll_store

        poll_store = create_poll_store(config)
        for district in poll_districts:
            poll_entries = poll_store.load_by_district(district["id"])
            if not poll_entries:
                continue
            poll_metas = [
                m for m in poll_store.load_meta()
                if m.district_id == district["id"]
            ]
            section = _format_poll_trend_table(poll_entries, poll_metas, district)
            if section:
                poll_sections.append(section)
                poll_data_for_chart.append((poll_entries, poll_metas, district))
                logger.info(
                    "여론조사 데이터 포함 — %s: %d건", district["name"], len(poll_entries),
                )

    if not results and not poll_sections:
        print("\n  검색 결과가 없습니다. 수집 파이프라인을 먼저 실행하세요.\n")
        return

    for entries, metas, dist in poll_data_for_chart:
        _print_poll_chart(entries, metas, dist)

    if results:
        _print_chunks(results, {"name": "전체"})

    scorer = _build_scorer(config)
    user_prompt = _build_qa_prompt(query, results, poll_sections=poll_sections)

    logger.info(
        "LLM 질의 응답 요청 — 청크 %d건, 여론조사 %d개 선거구",
        len(results), len(poll_sections),
    )
    t0 = time.monotonic()
    answer = scorer._call_llm(QA_SYSTEM_PROMPT, user_prompt, json_mode=False)
    elapsed = time.monotonic() - t0
    logger.info("LLM 응답 수신 — %.1f초", elapsed)

    _print_qa_answer(query, answer, results)


def _print_diagnosis(verdict) -> None:
    if not verdict.diagnosis:
        return
    print()
    print("=" * 60)
    print(f"🔍 문제점 진단: {verdict.district_name}")
    print("=" * 60)
    for d in verdict.diagnosis:
        print(f"\n  📌 {d.candidate} ({d.party}) — 심각도: {d.severity}")
        for i, p in enumerate(d.problems, 1):
            print(f"     문제 {i}: {p}")
        for i, r in enumerate(d.root_causes, 1):
            print(f"     원인 {i}: {r}")
        print(f"     종합: {d.summary}")
    print("=" * 60)


def _print_strategy(verdict) -> None:
    if not verdict.strategy:
        return
    print()
    print("=" * 60)
    print(f"🎯 대응방안/전략: {verdict.district_name}")
    print("=" * 60)
    for s in verdict.strategy:
        print(f"\n  📌 {s.candidate} ({s.party}) — 우선순위: {s.priority}")
        for i, sol in enumerate(s.solutions, 1):
            print(f"     방안 {i}: {sol}")
        print(f"     실행 계획:")
        for ap in s.action_plan:
            print(f"       • {ap}")
        print(f"     예상 효과: {s.expected_impact}")
        print(f"     종합: {s.summary}")
    print("=" * 60)


def _print_opinion_polls(verdict) -> None:
    pa = verdict.poll_analysis
    if not pa:
        return
    print()
    print("=" * 60)
    print(f"📊 여론조사 동향 분석: {verdict.district_name}")
    print(f"   분석 기간: {pa.analysis_period} | 총 {pa.total_surveys}회 조사")
    print("=" * 60)

    if pa.candidate_trends:
        print("\n  📈 후보별 추이")
        print("-" * 40)
        for ct in pa.candidate_trends:
            arrow = {"상승": "↑", "하락": "↓", "정체": "→"}.get(ct.trend_direction, "?")
            print(f"\n  {ct.candidate} ({ct.party})")
            print(f"     최신 지지율: {ct.latest_support}% {arrow} {ct.trend_direction}")
            print(f"     {ct.trend_description}")

    if pa.method_analysis:
        print("\n  🔬 조사 방법론 분석")
        print("-" * 40)
        for ma in pa.method_analysis:
            print(f"\n  [{ma.method}]")
            print(f"     특성: {ma.characteristics}")
            print(f"     신뢰성: {ma.reliability_note}")
            print(f"     결과: {ma.results_summary}")

    if pa.key_findings:
        print("\n  💡 핵심 발견")
        print("-" * 40)
        for i, finding in enumerate(pa.key_findings, 1):
            print(f"  {i}. {finding}")

    print(f"\n  📝 종합 추이: {pa.trend_summary}")
    print(f"\n  🔍 신뢰성 평가: {pa.reliability_assessment}")
    print("=" * 60)
    print()


def _print_comparison(verdict) -> None:
    if not verdict.comparison:
        return
    print()
    print("=" * 60)
    print(f"⚖️ 후보 비교 분석: {verdict.district_name}")
    print("=" * 60)
    for comp in verdict.comparison:
        print(f"\n  {comp.candidate_a} vs {comp.candidate_b}")
        print("-" * 40)
        for dim in comp.dimensions:
            print(f"\n  📊 {dim.dimension} — 우위: {dim.advantage}")
            print(f"     {comp.candidate_a}: {dim.candidate_a_assessment}")
            print(f"     {comp.candidate_b}: {dim.candidate_b_assessment}")
        print(f"\n  종합: {comp.overall_edge}")
        print(f"  요약: {comp.summary}")
    print("=" * 60)


def _run_opinion_polls_mode(args, config: dict, district: dict) -> None:
    """여론조사 분석 모드: PollStore 데이터 기반 동향 분석."""
    from rag.poll_store import create_poll_store

    logger.info("여론조사 데이터 로드 중…")
    poll_store = create_poll_store(config)
    poll_entries = poll_store.load_by_district(district["id"])
    poll_metas = [
        m for m in poll_store.load_meta()
        if m.district_id == district["id"]
    ]

    if not poll_entries:
        logger.error("선거구 '%s'에 여론조사 데이터가 없습니다.", district["id"])
        print(f"\n  여론조사 데이터가 없습니다. ({district['name']})")
        print("  polls.jsonl 또는 Google Sheets에 데이터를 추가하세요.\n")
        return

    logger.info(
        "여론조사 데이터 — %s: %d건 (메타 %d건)",
        district["name"], len(poll_entries), len(poll_metas),
    )

    scorer = _build_scorer(config)

    if args.no_graph:
        verdict = scorer.analyze_polls(poll_entries, poll_metas, district)
    else:
        from rag.verdict_graph import run_verdict_graph
        verdict = run_verdict_graph(
            scorer, [], district,
            mode="opinion_polls",
            poll_entries=poll_entries,
            poll_metas=poll_metas,
        )

    from rag.verdict_store import VerdictStore
    store = VerdictStore()
    store.save(verdict)

    _print_verdict(verdict)
    _print_opinion_polls(verdict)


def _run_verdict_mode(args, config: dict) -> None:
    """판정 모드: 선거구별 판세 분석."""
    district = _find_district(config, args.district)
    if not district:
        available = [d["id"] for d in config.get("districts", [])]
        logger.error("선거구 '%s' 없음. 사용 가능: %s", args.district, available)
        return

    analysis_mode = getattr(args, "analysis", "verdict") or "verdict"
    target_candidates = None
    if getattr(args, "candidates", None):
        target_candidates = [c.strip() for c in args.candidates.split(",")]

    if analysis_mode == "comparison" and (not target_candidates or len(target_candidates) < 2):
        candidate_names = [c["name"] for c in district.get("candidates", [])]
        logger.error(
            "comparison 모드에는 --candidates로 2명 이상 지정 필요. 후보: %s",
            ", ".join(candidate_names),
        )
        return

    if analysis_mode == "opinion_polls":
        _run_opinion_polls_mode(args, config, district)
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

    if args.no_graph:
        verdict = scorer.score(flat_chunks, district, grouped_chunks=grouped)
    else:
        from rag.verdict_graph import run_verdict_graph
        verdict = run_verdict_graph(
            scorer, flat_chunks, district, grouped_chunks=grouped,
            mode=analysis_mode, target_candidates=target_candidates,
        )

    from rag.verdict_store import VerdictStore
    store = VerdictStore()
    store.save(verdict)

    _print_verdict(verdict)

    if verdict.diagnosis:
        _print_diagnosis(verdict)
    if verdict.strategy:
        _print_strategy(verdict)
    if verdict.comparison:
        _print_comparison(verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Election Radar RAG 판정 파이프라인")
    parser.add_argument("--district", default=None, help="선거구 ID (pyeongtaek_b | busan_bukgu_gap)")
    parser.add_argument("--query", default=None, help="단발 질의 모드 — 자유 질문")
    parser.add_argument(
        "--analysis", default="verdict",
        choices=["verdict", "diagnosis", "strategy", "comparison", "opinion_polls"],
        help="분석 모드 (verdict: 판세, diagnosis: 문제점, strategy: 전략, comparison: 비교, opinion_polls: 여론조사 동향)",
    )
    parser.add_argument("--candidates", default=None, help="대상 후보 (쉼표 구분, comparison 모드 필수)")
    parser.add_argument("--top-k", type=int, default=None, help="검색 수 (config 우선)")
    parser.add_argument("--min-score", type=float, default=None, help="유사도 임계값 (config 우선)")
    parser.add_argument("--lookback-days", type=int, default=None, help="최근 N일 기사만 검색 (config 우선)")
    parser.add_argument("--purge-days", type=int, default=None, help="N일 이전 벡터 삭제")
    parser.add_argument("--skip-score", action="store_true", help="LLM 판정 생략 (검색 결과만)")
    parser.add_argument("--no-graph", action="store_true", help="LangGraph 비활성화 (기존 단발 호출)")
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
