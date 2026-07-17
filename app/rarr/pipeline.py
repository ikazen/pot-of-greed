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

    # M3: score 내림차순 정렬 후 dedup — 먼저 등장한(최고점) ref가 대표로 남게 해
    # 최종 source 목록이 claim 처리 순서가 아니라 관련도순이 되게 한다.
    all_evidence = sorted(
        (ev for r in reports for ev in r.evidence),
        key=lambda ev: ev.score,
        reverse=True,
    )

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

    agreement = await check_agreement(claim, evidence, deadline=deadline)
    if has_hallucination:
        agreement.agree = False

    revised_text, used_evidence, corrections = await edit_claim(
        claim, agreement, evidence, max_evidence=settings.rerank_top_k, deadline=deadline
    )
    hallucinated_refs = [ref for ref, exists in citation_map.items() if not exists]

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

    # M2: "무언가 정정됨"이 아니라 "원래 환각 ref가 최종 텍스트에서 실제로 사라졌는가"로 판정.
    # 이전 산식(bool(corrections) or ...)은 무관한 다른 교정만 있어도 환각까지 고쳐진 것으로
    # 오산해 hallucination_correction_rate를 부풀렸다.
    corrected = bool(hallucinated_refs) and all(
        ref not in revised_text for ref in hallucinated_refs
    )

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

    # M5: draft 실패는 그 자체로 답변 불가 상태이므로 별도 처리. 아래 파이프라인
    # 단계 실패 시의 degrade 경로가 이 draft_text를 재사용해 draft를 두 번 호출하지 않는다.
    try:
        draft_text = await draft(query)
    except Exception:
        logger.warning("RARR draft 실패", exc_info=True)
        return RarrResult(
            answer="답변을 생성할 수 없습니다.\n\n[미검증] 답변 검증에 실패했습니다. 내용을 반드시 확인하세요.",
            sources=[],
            warnings=[],
            attributions=[],
        )

    try:
        if time.monotonic() > deadline:
            raise TimeoutError("draft exceeded deadline")

        claims = await decompose_claims(draft_text)
        cap = settings.rarr_max_claims
        verified_claims = claims[:cap] if cap else claims
        deferred_claims = claims[cap:] if cap else []

        semaphore = asyncio.Semaphore(settings.rarr_max_concurrency)

        async def _bounded_process(c: Claim) -> AttributionReport:
            async with semaphore:
                return await _process_claim(c, mode, settings, deadline)

        reports = await asyncio.gather(
            *[_bounded_process(c) for c in verified_claims]
        )

        # 표시 답변은 원본 draft(마크다운 보존). claim 재조립(revised_text)은
        # metrics/attribution 내부용일 뿐 표시에는 쓰지 않는다 — 평문 원자 주장을
        # 공백으로 이어붙이면 draft의 제목/목록/개행이 전부 사라진다.
        # 환각으로 확인된 ref만 원본 draft에서 결정론적으로 스크럽.
        final_answer = draft_text
        hallucinated_all = {ref for r in reports for ref in r.hallucinated_refs}
        for ref in hallucinated_all:
            final_answer = final_answer.replace(ref, "[인용 삭제]")

        sources, warnings = _build_outputs(list(reports), limit=settings.source_top_k)

        # 근거 없는 claim은 문장마다 인라인 표식 대신 집계 경고 1개로.
        unverified = [r for r in reports if not r.evidence]
        if unverified:
            warnings.append(Warning(
                chunk_id="",
                ref="",
                validity_flag="unverified",
                message=f"[주의] {len(unverified)}/{len(reports)}개 문장이 코퍼스 근거로 확인되지 않았습니다. 내용을 반드시 확인하세요.",
            ))

        # H3: cap에 걸려 검증 못 한 claim은 draft 본문에 이미 포함돼 있으므로
        # 인라인 배너 없이 경고만 남긴다.
        if deferred_claims:
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
        return RarrResult(
            answer=draft_text + "\n\n[미검증] 답변 검증에 실패했습니다. 내용을 반드시 확인하세요.",
            sources=[],
            warnings=[],
            attributions=[],
        )
