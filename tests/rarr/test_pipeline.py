from __future__ import annotations

import pytest

from app.rarr.pipeline import RarrResult
from app.rarr.types import AttributionReport, Claim, Evidence


def _make_evidence(chunk_id="c1", validity_flag="valid"):
    return Evidence(
        chunk_id=chunk_id,
        ref="소득세법 제89조",
        text="관련 조문 내용",
        score=0.9,
        meta={"law_name": "소득세법", "article_no": "제89조", "validity_flag": validity_flag},
    )


def _noop_run_rarr_parts(monkeypatch):
    """pipeline 단계별 함수를 간단한 가짜로 교체."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안 텍스트"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="주장1"), Claim(text="주장2")]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)


@pytest.mark.asyncio
async def test_run_rarr_returns_rarr_result(monkeypatch):
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert isinstance(result, RarrResult)
    assert result.answer
    assert isinstance(result.sources, list)
    assert isinstance(result.warnings, list)


@pytest.mark.asyncio
async def test_run_rarr_preserves_draft_markdown(monkeypatch):
    """표시 답변은 claim 재조립이 아니라 원본 draft 그대로여야 한다(마크다운 보존)."""
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert result.answer == "초안 텍스트"
    assert "주장1" not in result.answer
    assert "주장2" not in result.answer


@pytest.mark.asyncio
async def test_run_rarr_degrade_on_failure(monkeypatch):
    import app.rarr.pipeline as pipeline_mod

    async def failing_draft(query, timeout=None):
        raise RuntimeError("LLM error")

    monkeypatch.setattr(pipeline_mod, "draft", failing_draft)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert "[미검증]" in result.answer
    assert result.sources == []


@pytest.mark.asyncio
async def test_run_rarr_builds_warnings_from_validity_flag(monkeypatch):
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="주장")]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence(validity_flag="overruled")]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert any(w.validity_flag == "overruled" for w in result.warnings)


@pytest.mark.asyncio
async def test_run_rarr_tracks_hallucinated_refs(monkeypatch):
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        from app.rarr.types import Claim
        return [Claim(text="소득세법 제999조 주장", cited_refs=["소득세법 제999조"])]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {"소득세법 제999조": False}  # 할루시네이션

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=False, supporting=[])

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return "수정된 주장 [정정: 제999조 → 제89조]", evidence, ["[정정: 제999조 → 제89조]"]

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    report = result.attributions[0]
    assert "소득세법 제999조" in report.hallucinated_refs
    assert report.corrected is True


@pytest.mark.asyncio
async def test_run_rarr_max_claims_cap(monkeypatch):
    """rarr_max_claims=2이면 decompose가 4개 반환해도 2개만 처리된다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text=f"주장{i}") for i in range(4)]

    processed_claims = []

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        processed_claims.append(claim.text)
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr

    settings = get_settings()
    monkeypatch.setattr(settings, "rarr_max_claims", 2)

    result = await run_rarr("질의", "simple", settings)
    assert len(result.attributions) == 2
    assert len(processed_claims) == 2


@pytest.mark.asyncio
async def test_run_rarr_max_claims_cap_marks_deferred(monkeypatch):
    """H3: cap 초과분은 draft 본문에 그대로 남아있고(삭제 없음) warning으로 표식된다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text=f"주장{i}") for i in range(4)]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr

    settings = get_settings()
    monkeypatch.setattr(settings, "rarr_max_claims", 2)

    result = await run_rarr("질의", "simple", settings)
    assert result.answer == "초안"  # deferred claim도 draft 본문에 이미 포함, 인라인 배너 없음
    assert any(w.validity_flag == "deferred" for w in result.warnings)


@pytest.mark.asyncio
async def test_run_rarr_no_cap_no_deferred_marker(monkeypatch):
    """cap 미설정(기본 0)이면 deferred 경로가 발동하지 않는다(회귀)."""
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert not any(w.validity_flag == "deferred" for w in result.warnings)


# ---------------------------------------------------------------------------
# C2/C3 — edit 후 ref 재검증 + 환각 ref 결정론적 제거·경고
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_rarr_removes_hallucination_newly_introduced_by_edit(monkeypatch):
    """edit LLM이 코퍼스에 없는 새 판례를 삽입해도 재검증되어 제거·경고돼야 한다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="소득세법 제89조에 따라 과세된다.", cited_refs=["소득세법 제89조"])]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        # 실재 코퍼스: 소득세법 제89조만. 나머지(edit가 심은 판례 등)는 전부 미검증.
        return {ref: (ref == "소득세법 제89조") for ref in refs}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=False, supporting=[], reason="edit 필요")

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        # edit가 근거 없는 새 판례번호를 만들어냄 (환각)
        return "소득세법 제89조 및 2099두99999 판례에 따라 과세된다.", evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    report = result.attributions[0]
    assert report.removed_refs == ["2099두99999"]
    assert "2099두99999" not in report.revised_text
    assert "[인용 삭제]" in report.revised_text
    assert "소득세법 제89조" in report.revised_text  # 실재 인용은 보존
    assert any(w.validity_flag == "hallucination" and w.ref == "2099두99999" for w in result.warnings)


@pytest.mark.asyncio
async def test_run_rarr_removes_hallucination_edit_failed_to_correct(monkeypatch):
    """edit가 [정정:]을 못 붙이고 환각 ref를 그대로 남겨도 안전망이 제거·경고한다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="소득세법 제999조 주장", cited_refs=["소득세법 제999조"])]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {ref: False for ref in refs}  # 전부 미실재

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=False, supporting=[])

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        # [정정:] 없이 원문 그대로 반환 (edit가 못 고침)
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    report = result.attributions[0]
    assert report.removed_refs == ["소득세법 제999조"]
    assert "소득세법 제999조" not in report.revised_text
    assert "[인용 삭제]" in report.revised_text
    assert any(w.validity_flag == "hallucination" for w in result.warnings)


# ---------------------------------------------------------------------------
# H1 — deadline이 agreement/edit까지 전파된다
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_claim_deadline_exceeded_degrades_without_llm_calls(monkeypatch):
    """deadline을 과거로 주면 check_agreement/edit_claim에 전달되어 LLM 호출 없이 원문 유지."""
    import time
    from app.rarr.pipeline import _process_claim
    from app.rarr.agreement import AgreementResult

    agreement_calls = []
    edit_calls = []

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    async def real_check_agreement(claim, evidence, deadline=None):
        agreement_calls.append(deadline)
        # 실제 check_agreement과 동일하게 deadline 초과 시 degrade
        if deadline is not None and deadline - time.monotonic() <= 0:
            return AgreementResult(agree=False, reason="deadline 초과 — 원문 유지")
        return AgreementResult(agree=True, supporting=evidence)

    async def real_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        edit_calls.append(deadline)
        if agreement.agree:
            return claim.text, agreement.supporting, []
        if deadline is not None and deadline - time.monotonic() <= 0:
            return claim.text, [], []
        return claim.text, evidence, []

    import app.rarr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", real_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", real_edit_claim)

    from app.config import get_settings
    past_deadline = time.monotonic() - 1
    report = await _process_claim(Claim(text="주장"), "simple", get_settings(), past_deadline)

    assert report.revised_text == "주장"
    assert agreement_calls == [past_deadline]
    assert edit_calls == [past_deadline]


@pytest.mark.asyncio
async def test_run_rarr_no_hallucination_no_removal(monkeypatch):
    """환각 없는 해피패스는 removed_refs 비어있고 텍스트·경고에 영향 없어야 한다(회귀)."""
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    for report in result.attributions:
        assert report.removed_refs == []
    assert not any(w.validity_flag == "hallucination" for w in result.warnings)


@pytest.mark.asyncio
async def test_run_rarr_scrubs_hallucinated_ref_from_draft_answer(monkeypatch):
    """draft 원문에 실린 환각 ref는 표시 답변에서 [인용 삭제]로 치환되고, 실재 ref는 보존된다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "소득세법 제89조 및 2099두99999 판례에 따라 과세된다."

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(
            text="소득세법 제89조 및 2099두99999 판례에 따라 과세된다.",
            cited_refs=["소득세법 제89조", "2099두99999"],
        )]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {ref: (ref == "소득세법 제89조") for ref in refs}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    assert "2099두99999" not in result.answer
    assert "[인용 삭제]" in result.answer
    assert "소득세법 제89조" in result.answer


@pytest.mark.asyncio
async def test_run_rarr_unverified_claims_get_aggregate_warning(monkeypatch):
    """근거 없는 claim은 문장마다 인라인 표식 대신 집계 경고 1개로 노출된다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안 텍스트"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="주장1"), Claim(text="주장2")]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return []  # 근거 전무

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=False, supporting=[])

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text + " [미검증]", [], []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    assert result.answer == "초안 텍스트"
    assert "[미검증]" not in result.answer
    unverified_warnings = [w for w in result.warnings if w.validity_flag == "unverified"]
    assert len(unverified_warnings) == 1
    assert "2/2" in unverified_warnings[0].message


# ---------------------------------------------------------------------------
# BON-222 — M1-M5 + LOW 품질 개선 묶음
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_claim_corrected_requires_hallucinated_ref_removed_from_final_text(monkeypatch):
    """M2: 무관한 교정([정정:])만 있고 환각 ref가 최종 텍스트에 그대로 남으면 corrected=False.

    이전 산식(bool(corrections) or ...)은 다른 부분 교정만 있어도 "그 환각이
    정정됐다"고 오판해 hallucination_correction_rate를 부풀렸다.
    """
    import time
    import app.rarr.pipeline as pipeline_mod
    from app.rarr.pipeline import _process_claim
    from app.rarr.agreement import AgreementResult

    claim = Claim(text="소득세법 제999조 주장", cited_refs=["소득세법 제999조"])

    async def fake_research_claim(c, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    verify_calls = []

    async def fake_verify_citations(refs):
        verify_calls.append(list(refs))
        # 최초 호출(citation_map)은 미실재로, 재검증 호출은 (불일치하게) 실재로 답해
        # C3 제거가 발동하지 않는 상황을 재현 — 그래도 M2 산식은 최종 텍스트에
        # 남은 원래 환각 ref 문자열 자체를 직접 확인해야 한다.
        if len(verify_calls) == 1:
            return {ref: False for ref in refs}
        return {ref: True for ref in refs}

    async def fake_check_agreement(c, evidence, deadline=None):
        return AgreementResult(agree=False, supporting=[])

    async def fake_edit_claim(c, agreement, evidence, max_evidence=5, deadline=None):
        return "소득세법 제999조 주장 [정정: 오타 수정]", evidence, ["[정정: 오타 수정]"]

    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    deadline = time.monotonic() + 20
    report = await _process_claim(claim, "simple", get_settings(), deadline)

    assert "소득세법 제999조" in report.hallucinated_refs
    assert "소득세법 제999조" in report.revised_text
    assert report.corrected is False


@pytest.mark.asyncio
async def test_run_rarr_sources_sorted_by_score_descending(monkeypatch):
    """M3: source 목록이 claim 처리 순서가 아니라 evidence score 내림차순으로 정렬된다."""
    import app.rarr.pipeline as pipeline_mod

    def _scored_evidence(chunk_id, score):
        return Evidence(
            chunk_id=chunk_id,
            ref=f"소득세법 제{chunk_id}조",
            text="내용",
            score=score,
            meta={"law_name": "소득세법", "article_no": f"제{chunk_id}조"},
        )

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="주장1"), Claim(text="주장2")]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        if claim.text == "주장1":
            return [_scored_evidence("low", 0.3)]
        return [_scored_evidence("high", 0.95)]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    assert [s.chunk_id for s in result.sources] == ["high", "low"]


@pytest.mark.asyncio
async def test_run_rarr_bounds_claim_concurrency(monkeypatch):
    """M4: rarr_max_concurrency로 동시에 처리되는 claim 수가 제한된다."""
    import asyncio
    import app.rarr.pipeline as pipeline_mod
    from app.rarr.types import AttributionReport

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text=f"주장{i}") for i in range(6)]

    current = 0
    peak = 0

    async def fake_process_claim(claim, mode, settings, deadline, search_semaphore=None):
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return AttributionReport(claim=claim, revised_text=claim.text)

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "_process_claim", fake_process_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    settings = get_settings()
    monkeypatch.setattr(settings, "rarr_max_concurrency", 2)

    await run_rarr("질의", "simple", settings)
    assert peak <= 2


@pytest.mark.asyncio
async def test_run_rarr_bounds_total_search_concurrency_across_claims(monkeypatch):
    """#15: claim 세마포어와 search 세마포어가 중첩되지 않고, 전체 서브쿼리 검색
    동시 발사 수가 claim 수와 무관하게 rarr_max_concurrency로 캡핑된다.

    이전 버그: claim 세마포어(N) x _research_complex가 claim마다 새로 만드는
    question 세마포어(N)가 중첩돼 최악 N^2. 이 테스트는 4 claim x 3 question(=12
    총 검색)을 rarr_max_concurrency=2로 돌려 실제 동시 검색 피크가 2를 넘지
    않는지(구버전이라면 최악 4까지 가능) 확인한다.
    """
    import asyncio
    import app.rarr.pipeline as pipeline_mod
    from app.rarr.agreement import AgreementResult

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text=f"주장{i}") for i in range(4)]

    async def fake_verify_citations(refs):
        return {}

    async def fake_check_agreement(claim, evidence, deadline=None):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        return claim.text, evidence, []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    async def fake_generate_questions(claim, deadline=None):
        return ["q1", "q2", "q3"]

    current = 0
    peak = 0

    async def fake_search_complex(query, settings):
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return []

    async def fake_rerank(query, chunks, top_k):
        return []

    async def fake_expand_2hop(ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    import app.rarr.query_gen as qg_mod
    import app.api.chat as chat_mod
    from app.retrieval import reranker as reranker_mod
    from app.retrieval import graph_expand as ge_mod
    from app.retrieval import context_expand as ce_mod

    monkeypatch.setattr(qg_mod, "generate_questions", fake_generate_questions)
    monkeypatch.setattr(chat_mod, "_search_complex", fake_search_complex)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)
    monkeypatch.setattr(ge_mod, "expand_2hop", fake_expand_2hop)
    monkeypatch.setattr(ce_mod, "expand_to_parents", fake_expand_to_parents)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    settings = get_settings()
    monkeypatch.setattr(settings, "rarr_max_concurrency", 2)

    await run_rarr("질의", "complex", settings)
    assert peak <= 2


@pytest.mark.asyncio
async def test_run_rarr_draft_called_once_on_pipeline_failure(monkeypatch):
    """M5: draft 성공 후 후속 단계가 실패해도 draft를 두 번 호출하지 않고 결과를 재사용한다."""
    import app.rarr.pipeline as pipeline_mod

    draft_calls = []

    async def fake_draft(query, timeout=None):
        draft_calls.append(query)
        return "초안 텍스트"

    async def failing_decompose(text):
        raise RuntimeError("decompose 실패")

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", failing_decompose)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())

    assert len(draft_calls) == 1
    assert "초안 텍스트" in result.answer
    assert "[미검증]" in result.answer


# ---------------------------------------------------------------------------
# #14 — simple/complex 모드별 검증 deadline 예산 분리
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_rarr_uses_simple_mode_timeout_for_simple(monkeypatch):
    import time
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    captured_deadline = {}

    async def fake_decompose_claims(text, deadline=None):
        captured_deadline["value"] = deadline
        return []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    settings = get_settings()
    monkeypatch.setattr(settings, "simple_mode_timeout_s", 4)
    monkeypatch.setattr(settings, "complex_mode_timeout_s", 20)

    t0 = time.monotonic()
    await run_rarr("질의", "simple", settings)

    remaining = captured_deadline["value"] - t0
    assert 3.5 <= remaining <= 4.5  # simple 예산(4s) 근처, complex(20s)와는 뚜렷이 구분


@pytest.mark.asyncio
async def test_run_rarr_uses_complex_mode_timeout_for_complex(monkeypatch):
    import time
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query, timeout=None):
        return "초안"

    captured_deadline = {}

    async def fake_decompose_claims(text, deadline=None):
        captured_deadline["value"] = deadline
        return []

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    settings = get_settings()
    monkeypatch.setattr(settings, "simple_mode_timeout_s", 4)
    monkeypatch.setattr(settings, "complex_mode_timeout_s", 20)

    t0 = time.monotonic()
    await run_rarr("질의", "complex", settings)

    remaining = captured_deadline["value"] - t0
    assert 19.5 <= remaining <= 20.5


# ---------------------------------------------------------------------------
# #13 — on_progress 콜백 (draft 완료 / 분해 완료 / claim별 검증 진행)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_rarr_on_progress_reports_draft_decompose_and_claims(monkeypatch):
    _noop_run_rarr_parts(monkeypatch)  # fake_decompose_claims는 주장 2개 반환

    events: list[str] = []

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    await run_rarr("질의", "simple", get_settings(), on_progress=events.append)

    assert events[0] == "초안 작성 완료"
    assert events[1] == "2개 주장 분해 완료"
    # claim 2개 처리 완료 이벤트가 순서대로(1/2, 2/2) 뒤따라야 함
    assert "검증 1/2" in events
    assert "검증 2/2" in events
    assert events.index("검증 1/2") < events.index("검증 2/2")


@pytest.mark.asyncio
async def test_run_rarr_no_on_progress_is_noop(monkeypatch):
    """on_progress 미전달 시 아무 콜백도 안 걸리고 정상 동작(하위호환)."""
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert isinstance(result, RarrResult)


@pytest.mark.asyncio
async def test_run_rarr_on_progress_not_called_on_draft_failure(monkeypatch):
    import app.rarr.pipeline as pipeline_mod

    async def failing_draft(query, timeout=None):
        raise RuntimeError("LLM error")

    monkeypatch.setattr(pipeline_mod, "draft", failing_draft)

    events: list[str] = []

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    await run_rarr("질의", "simple", get_settings(), on_progress=events.append)

    assert events == []


# ---------------------------------------------------------------------------
# #33 — 단계별 시간 로깅
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_rarr_logs_stage_timings_with_consistent_run_id(monkeypatch, caplog):
    """draft/decompose/claim(x2)/total 로그가 전부 찍히고 run_id가 전 라인에서 동일해야 한다."""
    import logging
    _noop_run_rarr_parts(monkeypatch)  # fake_decompose_claims는 주장 2개 반환

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr

    with caplog.at_level(logging.INFO, logger="app.rarr.pipeline"):
        await run_rarr("질의", "simple", get_settings())

    rarr_logs = [r.message for r in caplog.records if r.message.startswith("RARR stage=")]
    stages = [line.split("stage=")[1].split(" ")[0] for line in rarr_logs]

    assert stages.count("draft") == 1
    assert stages.count("decompose") == 1
    assert stages.count("claim") == 2  # fake_decompose_claims가 2개 반환
    assert stages.count("total") == 1
    assert stages[-1] == "total"  # 총 소요시간 로그가 마지막

    run_ids = {line.split("run_id=")[1].split(" ")[0] for line in rarr_logs}
    assert len(run_ids) == 1  # 전 라인이 같은 run_id

    total_line = next(line for line in rarr_logs if "stage=total" in line)
    assert "outcome=success" in total_line
    assert "elapsed_ms=" in total_line

    claim_lines = [line for line in rarr_logs if "stage=claim" in line]
    for line in claim_lines:
        assert "research_ms=" in line
        assert "agreement_ms=" in line
        assert "edit_ms=" in line
        assert "total_ms=" in line


# ---------------------------------------------------------------------------
# 디버그 모드 — settings.debug_pipeline ON일 때만 RarrResult.debug 채워짐
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_rarr_debug_off_by_default(monkeypatch):
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert result.debug is None


@pytest.mark.asyncio
async def test_run_rarr_debug_on_includes_claim_trace(monkeypatch):
    """debug_pipeline=True면 주장별 원문/판정/수정 내역이 채워지고, 변경 없는
    주장의 revised는 None(diff 요약 원칙: 바뀐 것만 표시)이어야 한다."""
    import app.rarr.pipeline as pipeline_mod
    from app.rarr.agreement import AgreementResult

    async def fake_draft(query, timeout=None):
        return "초안"

    async def fake_decompose_claims(text, deadline=None):
        return [Claim(text="주장1"), Claim(text="소득세법 제999조 주장", cited_refs=["소득세법 제999조"])]

    async def fake_research_claim(claim, mode, settings, deadline, search_semaphore=None):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {ref: False for ref in refs}

    async def fake_check_agreement(claim, evidence, deadline=None):
        if claim.text == "주장1":
            return AgreementResult(agree=True, supporting=evidence, reason="근거 일치")
        return AgreementResult(agree=False, supporting=[], reason="근거 불일치")

    async def fake_edit_claim(claim, agreement, evidence, max_evidence=5, deadline=None):
        if agreement.agree:
            return claim.text, evidence, []
        return "수정된 주장 [정정: 제999조 → 제89조]", evidence, ["[정정: 제999조 → 제89조]"]

    monkeypatch.setattr(pipeline_mod, "draft", fake_draft)
    monkeypatch.setattr(pipeline_mod, "decompose_claims", fake_decompose_claims)
    monkeypatch.setattr(pipeline_mod, "research_claim", fake_research_claim)
    monkeypatch.setattr(pipeline_mod, "verify_citations", fake_verify_citations)
    monkeypatch.setattr(pipeline_mod, "check_agreement", fake_check_agreement)
    monkeypatch.setattr(pipeline_mod, "edit_claim", fake_edit_claim)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    settings = get_settings()
    monkeypatch.setattr(settings, "debug_pipeline", True)

    result = await run_rarr("질의", "simple", settings)
    assert result.debug is not None
    assert result.debug["mode"] == "simple"
    assert result.debug["claims_total"] == 2

    claims = result.debug["claims"]
    unchanged = next(c for c in claims if c["original"] == "주장1")
    assert unchanged["agree"] is True
    assert unchanged["revised"] is None  # 변경 없으므로 None

    changed = next(c for c in claims if "제999조" in c["original"])
    assert changed["agree"] is False
    assert changed["agreement_reason"] == "근거 불일치"
    assert changed["revised"] == "수정된 주장 [정정: 제999조 → 제89조]"


@pytest.mark.asyncio
async def test_run_rarr_logs_total_outcome_draft_failed(monkeypatch, caplog):
    """draft 실패 시 stage=total outcome=draft_failed 하나만 찍히고 draft/decompose/claim은 없어야 한다."""
    import logging
    import app.rarr.pipeline as pipeline_mod

    async def failing_draft(query, timeout=None):
        raise RuntimeError("LLM error")

    monkeypatch.setattr(pipeline_mod, "draft", failing_draft)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr

    with caplog.at_level(logging.INFO, logger="app.rarr.pipeline"):
        await run_rarr("질의", "simple", get_settings())

    rarr_logs = [r.message for r in caplog.records if r.message.startswith("RARR stage=")]
    assert len(rarr_logs) == 1
    assert "stage=total" in rarr_logs[0]
    assert "outcome=draft_failed" in rarr_logs[0]
