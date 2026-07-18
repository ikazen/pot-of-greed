"""RARR 파이프라인 eval 하니스.

사용법:
  python -m scripts.rarr_eval --mode both --report
  python -m scripts.rarr_eval --mode simple --limit 4 --out result.json

인프라(pgvector+Neo4j+Gemini/Ollama Cloud) 미가동 시 RARR 안전망이
[미검증] degrade → 하니스는 degrade 비율도 집계하며 에러로 종료되지 않는다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# app.main을 거치지 않는 독립 스크립트라 #33의 basicConfig가 안 걸린다 — 이
# 하니스는 stage=draft/decompose/claim/total 단계별 로그(#33)가 핵심 산출물이므로
# 여기서 직접 활성화한다.
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _load_queries(path: Path, mode_filter: str, limit: int) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if mode_filter != "both":
        items = [q for q in items if q.get("mode") == mode_filter]
    if limit:
        items = items[:limit]
    return items


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return sorted_v[idx]


async def _run_one(query_item: dict, settings) -> dict:
    from app.rarr.metrics import compute_metrics
    from app.rarr.pipeline import run_rarr

    query = query_item["query"]
    mode = query_item.get("mode", "simple")
    tag = query_item.get("tag", "")

    t0 = perf_counter()
    result = await run_rarr(query, mode, settings)
    elapsed = perf_counter() - t0

    degraded = "[미검증] 답변 검증에 실패했습니다" in result.answer and not result.attributions
    metrics = compute_metrics(result.attributions)

    return {
        "query": query[:60],
        "mode": mode,
        "tag": tag,
        "elapsed_s": round(elapsed, 2),
        "degraded": degraded,
        "n_claims": metrics.n_claims,
        "attribution_score": round(metrics.attribution_score, 3),
        "preservation_score": round(metrics.preservation_score, 3),
        "n_hallucinated": metrics.n_hallucinated,
        "hallucination_correction_rate": round(metrics.hallucination_correction_rate, 3),
        "n_sources": len(result.sources),
        "n_warnings": len(result.warnings),
    }


def _aggregate(rows: list[dict]) -> dict:
    def _by_mode(mode: str) -> list[dict]:
        return [r for r in rows if r["mode"] == mode]

    agg: dict = {"total": len(rows), "by_mode": {}}
    for mode in ("simple", "complex"):
        subset = _by_mode(mode)
        if not subset:
            continue
        elapsed = [r["elapsed_s"] for r in subset]
        agg["by_mode"][mode] = {
            "n": len(subset),
            "degraded": sum(1 for r in subset if r["degraded"]),
            "elapsed_p50": round(_percentile(elapsed, 50), 2),
            "elapsed_p95": round(_percentile(elapsed, 95), 2),
            "attribution_score": round(statistics.mean(r["attribution_score"] for r in subset), 3),
            "preservation_score": round(statistics.mean(r["preservation_score"] for r in subset), 3),
            "n_hallucinated_total": sum(r["n_hallucinated"] for r in subset),
            "hallucination_correction_rate": round(
                statistics.mean(r["hallucination_correction_rate"] for r in subset), 3
            ),
        }
    return agg


def _print_table(rows: list[dict], agg: dict) -> None:
    header = f"{'query':<42} {'mode':<8} {'tag':<12} {'s':>5} {'attr':>5} {'pres':>5} {'hall':>5} {'corr':>5} {'deg':>4}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['query']:<42} {r['mode']:<8} {r['tag']:<12} "
            f"{r['elapsed_s']:>5.2f} {r['attribution_score']:>5.3f} "
            f"{r['preservation_score']:>5.3f} {r['n_hallucinated']:>5} "
            f"{r['hallucination_correction_rate']:>5.3f} {'Y' if r['degraded'] else 'N':>4}"
        )
    print()
    for mode, stats in agg["by_mode"].items():
        print(f"[{mode}] n={stats['n']} degraded={stats['degraded']} "
              f"p50={stats['elapsed_p50']}s p95={stats['elapsed_p95']}s "
              f"attr={stats['attribution_score']} pres={stats['preservation_score']} "
              f"hall={stats['n_hallucinated_total']} corr_rate={stats['hallucination_correction_rate']}")


def _write_report(rows: list[dict], agg: dict, settings) -> Path:
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{ts}.md"

    lines = [
        f"# RARR eval report — {ts}",
        "",
        "## 설정",
        f"- rarr_max_claims: {settings.rarr_max_claims}",
        f"- rarr_questions_per_claim: {settings.rarr_questions_per_claim}",
        f"- complex_mode_timeout_s: {settings.complex_mode_timeout_s}",
        "",
        "## 집계",
    ]
    for mode, stats in agg["by_mode"].items():
        lines += [
            f"### {mode}",
            f"- n: {stats['n']}, degraded: {stats['degraded']}",
            f"- 지연: p50={stats['elapsed_p50']}s, p95={stats['elapsed_p95']}s",
            f"- attribution: {stats['attribution_score']}",
            f"- preservation: {stats['preservation_score']}",
            f"- 할루시네이션 주장 수: {stats['n_hallucinated_total']}",
            f"- 교정율: {stats['hallucination_correction_rate']}",
            "",
        ]

    lines += ["## 개별 결과", ""]
    for r in rows:
        lines.append(
            f"- [{r['mode']}|{r['tag']}] {r['query']} "
            f"→ {r['elapsed_s']}s attr={r['attribution_score']} "
            f"hall={r['n_hallucinated']} corr={r['hallucination_correction_rate']}"
            + (" [DEGRADED]" if r["degraded"] else "")
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(description="RARR eval 하니스")
    parser.add_argument("--mode", choices=["simple", "complex", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="질의 수 제한 (0=전체)")
    parser.add_argument("--out", default=None, help="JSON 리포트 출력 경로")
    parser.add_argument("--report", action="store_true", help="eval/results/<ts>.md 리포트 저장")
    parser.add_argument(
        "--queries",
        default=None,
        help="질의셋 JSON 경로 (기본: scripts/eval_queries.json)",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries) if args.queries else Path(__file__).parent / "eval_queries.json"
    if not queries_path.exists():
        print(f"질의셋 파일 없음: {queries_path}", file=sys.stderr)
        sys.exit(1)

    from app.config import get_settings
    from app.db.pg import init_pg, close_pg
    from app.db.neo4j import init_neo4j, close_neo4j

    settings = get_settings()

    items = _load_queries(queries_path, args.mode, args.limit)
    if not items:
        print("실행할 질의 없음.")
        return

    print(f"실행: {len(items)}개 질의 (mode={args.mode})")
    print(f"노브: max_claims={settings.rarr_max_claims}, questions_per_claim={settings.rarr_questions_per_claim}")
    print()

    # PG/Neo4j 풀 초기화 — 이게 없으면 research 단계에서 "PG pool not initialised"
    # RuntimeError가 run_rarr의 최상위 except에 조용히 잡혀 매 질의가 degrade되고,
    # 하니스는 에러 없이 "정상 종료"돼 이 하니스가 사실상 아무것도 측정하지
    # 못하고 있었다(#35).
    await init_pg(settings.pg_dsn)
    await init_neo4j(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        rows = []
        for i, item in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] {item['query'][:50]}...", end=" ", flush=True)
            row = await _run_one(item, settings)
            rows.append(row)
            status = "DEGRADED" if row["degraded"] else f"{row['elapsed_s']}s"
            print(status)
    finally:
        await close_pg()
        await close_neo4j()

    print()
    agg = _aggregate(rows)
    _print_table(rows, agg)

    if args.out:
        out = {"rows": rows, "aggregate": agg}
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 저장: {args.out}")

    if args.report:
        report_path = _write_report(rows, agg, settings)
        print(f"리포트 저장: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
