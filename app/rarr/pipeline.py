from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.rarr.agreement import check_agreement
from app.rarr.citation import verify_citations
from app.rarr.claims import _extract_refs, decompose_claims
from app.rarr.draft import draft
from app.rarr.edit import edit_claim
from app.rarr.research import research_claim
from app.rarr.types import AttributionReport, Claim, Evidence
from app.reasoning.answer_builder import (
    Source,
    Warning,
    _VALIDITY_FLAGS,
    _build_warning_message,
    _first_line,
)

logger = logging.getLogger(__name__)


@dataclass
class RarrResult:
    answer: str
    sources: list[Source]
    warnings: list[Warning]
    attributions: list[AttributionReport]


def _evidence_type(ev: Evidence) -> str:
    return "article" if ev.meta.get("law_name") else "case"


def _build_outputs(
    reports: list[AttributionReport],
    limit: int,
) -> tuple[list[Source], list[Warning]]:
    """AttributionReport 목록 → Source/Warning 목록 (결정론)."""
    seen_refs: set[str] = set()
    sources: list[Source] = []
    warnings: list[Warning] = []

    all_evidence = [ev for r in reports for ev in r.evidence]

    for ev in all_evidence:
        if ev.ref in seen_refs:
            continue
        seen_refs.add(ev.ref)
        sources.append(Source(
            type=_evidence_type(ev),
            ref=ev.ref,
            chunk_id=ev.chunk_id,
            summary=_first_line(ev.text),
        ))
        flag = ev.meta.get("validity_flag", "")
        if flag in _VALIDITY_FLAGS:
            warnings.append(Warning(
                chunk_id=ev.chunk_id,
                ref=ev.ref,
                validity_flag=flag,
                message=_build_warning_message(flag, ev.meta),
            ))

    # corrections → 추가 warnings
    for r in reports:
        for corr in r.corrections:
            if corr:
                warnings.append(Warning(
                    chunk_id="",
                    ref=corr,
                    validity_flag="correction",
                    message=corr,
                ))

    # C3: 결정론적으로 제거된 미검증 ref → 경고 강제 (edit가 [정정:]을 안 붙여도 승격)
    for r in reports:
        for ref in r.removed_refs:
            warnings.append(Warning(
                chunk_id="",
                ref=ref,
                validity_flag="hallucination",
                message=f"[주의] '{ref}'는 코퍼스에서 확인되지 않아 제거되었습니다.",
            ))

    return sources[:limit], warnings


async def _process_claim(
    claim: Claim,
    mode: str,
    settings,
    deadline: float,
) -> AttributionReport:
    evidence = await research_claim(claim, mode, settings, deadline)
    citation_map = await verify_citations(claim.cited_refs)

    # 할루시네이션 인용이 있으면 agree 강제 False (agreement 앞 prune)
    has_hallucination = any(not exists for exists in citation_map.values())

    agreement = await check_agreement(claim, evidence)
    if has_hallucination:
        agreement.agree = False

    revised_text, used_evidence, corrections = await edit_claim(claim, agreement, evidence)
    hallucinated_refs = [ref for ref, exists in citation_map.items() if not exists]
    corrected = bool(corrections) or (revised_text != claim.text and bool(hallucinated_refs))

    # C2: edit 결과의 ref 재검증. edit가 그대로면(agree 경로) 이미 검증된 citation_map 재사용해
    # DB 재호출을 회피하고, 수정됐다면 edit LLM이 새로 심었거나 못 지운 환각까지 다시 잡는다.
    if revised_text == claim.text:
        revised_refs, revised_map = claim.cited_refs, citation_map
    else:
        revised_refs = _extract_refs(revised_text)
        revised_map = await verify_citations(revised_refs)

    # C3: 최종 텍스트에 남은 미검증 ref는 결정론적으로 제거 — LLM이 [정정:]을 안 붙여도 안전망 작동.
    removed_refs = [ref for ref in revised_refs if not revised_map.get(ref, False)]
    for ref in removed_refs:
        revised_text = revised_text.replace(ref, "[인용 삭제]")

    return AttributionReport(
        claim=claim,
        evidence=used_evidence,
        agree=agreement.agree,
        revised_text=revised_text,
        corrections=corrections,
        hallucinated_refs=hallucinated_refs,
        corrected=corrected,
        removed_refs=removed_refs,
    )


async def run_rarr(query: str, mode: str, settings) -> RarrResult:
    """RARR 파이프라인 전체 실행.

    실패·타임아웃 시 순수 초안 + [미검증] 배너로 degrade.
    """
    deadline = time.monotonic() + settings.complex_mode_timeout_s

    try:
        draft_text = await draft(query)

        if time.monotonic() > deadline:
            raise TimeoutError("draft exceeded deadline")

        claims = await decompose_claims(draft_text)
        cap = settings.rarr_max_claims
        verified_claims = claims[:cap] if cap else claims
        deferred_claims = claims[cap:] if cap else []

        reports = await asyncio.gather(
            *[_process_claim(c, mode, settings, deadline) for c in verified_claims]
        )

        final_answer = " ".join(r.revised_text for r in reports)

        sources, warnings = _build_outputs(list(reports), limit=settings.source_top_k)

        # H3: cap에 걸려 검증 못 한 claim은 원문 그대로 붙이되 미검증 표식.
        # 삭제 대신 유지 — 답변 내용 손실보다 "검증 안 됨"을 드러내는 쪽을 택함.
        if deferred_claims:
            deferred_text = " ".join(c.text for c in deferred_claims)
            final_answer += f"\n\n[미검증] 아래 항목은 검증 한도 초과로 확인되지 않았습니다:\n{deferred_text}"
            warnings.append(Warning(
                chunk_id="",
                ref="",
                validity_flag="deferred",
                message=f"[주의] {len(deferred_claims)}개 항목이 검증 한도(rarr_max_claims={cap})를 초과해 검증되지 않았습니다.",
            ))

        # 3층 법리 (complex 모드 + 경고 있을 때만)
        if mode == "complex" and warnings:
            from app.reasoning.answer_builder import legal_reasoning_layer
            from app.retrieval.vector_search import Chunk

            # Evidence → Chunk (legal_reasoning_layer 인터페이스 호환)
            chunks_for_reasoning: list[Chunk] = [
                Chunk(
                    chunk_id=ev.chunk_id,
                    table=_evidence_type(ev),
                    text=ev.text,
                    score=ev.score,
                    meta=ev.meta,
                )
                for r in reports for ev in r.evidence
            ]
            reasoning = await legal_reasoning_layer(query, chunks_for_reasoning, warnings)
            if reasoning:
                final_answer += f"\n\n## 법리 검토\n{reasoning}"

        return RarrResult(
            answer=final_answer,
            sources=sources,
            warnings=warnings,
            attributions=list(reports),
        )

    except Exception:
        logger.warning("RARR pipeline failed — degrading to draft", exc_info=True)
        try:
            draft_text = await draft(query)
        except Exception:
            draft_text = "답변을 생성할 수 없습니다."
        return RarrResult(
            answer=draft_text + "\n\n[미검증] 답변 검증에 실패했습니다. 내용을 반드시 확인하세요.",
            sources=[],
            warnings=[],
            attributions=[],
        )
