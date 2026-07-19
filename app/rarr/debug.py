from __future__ import annotations

from app.rarr.types import AttributionReport


def build_debug_trace(
    reports: list[AttributionReport],
    mode: str,
    deferred_count: int,
    hallucinated_all: set[str],
    legal_reasoning_applied: bool,
) -> dict:
    """RARR 단계별 수정 내역을 JSON 직렬화 가능한 dict로 요약.

    debug_pipeline 설정이 켜졌을 때만 호출된다(app.rarr.pipeline.run_rarr).
    gemini 초안 원문은 포함하지 않는다 — 초안은 '원본'이지 '수정'이 아니므로
    decompose된 각 주장을 시작점으로 삼는다.
    """
    return {
        "mode": mode,
        "claims_total": len(reports),
        "deferred_count": deferred_count,
        "legal_reasoning_applied": legal_reasoning_applied,
        "scrubbed_refs": sorted(hallucinated_all),
        "claims": [_claim_trace(r) for r in reports],
    }


def _claim_trace(report: AttributionReport) -> dict:
    revised = report.revised_text
    changed = revised != report.claim.text
    return {
        "original": report.claim.text,
        "agree": report.agree,
        "agreement_reason": report.agreement_reason,
        "revised": revised if changed else None,
        "corrections": report.corrections,
        "hallucinated_refs": report.hallucinated_refs,
        "removed_refs": report.removed_refs,
        "evidence_refs": [ev.ref for ev in report.evidence],
        "evidence_count": len(report.evidence),
    }
