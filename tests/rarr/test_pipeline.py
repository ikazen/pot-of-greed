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

    async def fake_draft(query):
        return "초안 텍스트"

    async def fake_decompose_claims(text):
        return [Claim(text="주장1"), Claim(text="주장2")]

    async def fake_research_claim(claim, mode, settings, deadline):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence):
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
async def test_run_rarr_reassembles_claims(monkeypatch):
    _noop_run_rarr_parts(monkeypatch)

    from app.config import get_settings
    from app.rarr.pipeline import run_rarr
    result = await run_rarr("질의", "simple", get_settings())
    assert "주장1" in result.answer
    assert "주장2" in result.answer


@pytest.mark.asyncio
async def test_run_rarr_degrade_on_failure(monkeypatch):
    import app.rarr.pipeline as pipeline_mod

    async def failing_draft(query):
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

    async def fake_draft(query):
        return "초안"

    async def fake_decompose_claims(text):
        return [Claim(text="주장")]

    async def fake_research_claim(claim, mode, settings, deadline):
        return [_make_evidence(validity_flag="overruled")]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence):
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

    async def fake_draft(query):
        return "초안"

    async def fake_decompose_claims(text):
        from app.rarr.types import Claim
        return [Claim(text="소득세법 제999조 주장", cited_refs=["소득세법 제999조"])]

    async def fake_research_claim(claim, mode, settings, deadline):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {"소득세법 제999조": False}  # 할루시네이션

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=False, supporting=[])

    async def fake_edit_claim(claim, agreement, evidence):
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

    async def fake_draft(query):
        return "초안"

    async def fake_decompose_claims(text):
        return [Claim(text=f"주장{i}") for i in range(4)]

    processed_claims = []

    async def fake_research_claim(claim, mode, settings, deadline):
        processed_claims.append(claim.text)
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence):
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
    """H3: cap 초과분은 삭제 대신 원문 유지 + 미검증 배너·warning으로 표식된다."""
    import app.rarr.pipeline as pipeline_mod

    async def fake_draft(query):
        return "초안"

    async def fake_decompose_claims(text):
        return [Claim(text=f"주장{i}") for i in range(4)]

    async def fake_research_claim(claim, mode, settings, deadline):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=True, supporting=evidence)

    async def fake_edit_claim(claim, agreement, evidence):
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
    assert "주장2" in result.answer
    assert "주장3" in result.answer
    assert "[미검증]" in result.answer
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

    async def fake_draft(query):
        return "초안"

    async def fake_decompose_claims(text):
        return [Claim(text="소득세법 제89조에 따라 과세된다.", cited_refs=["소득세법 제89조"])]

    async def fake_research_claim(claim, mode, settings, deadline):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        # 실재 코퍼스: 소득세법 제89조만. 나머지(edit가 심은 판례 등)는 전부 미검증.
        return {ref: (ref == "소득세법 제89조") for ref in refs}

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=False, supporting=[], reason="edit 필요")

    async def fake_edit_claim(claim, agreement, evidence):
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

    async def fake_draft(query):
        return "초안"

    async def fake_decompose_claims(text):
        return [Claim(text="소득세법 제999조 주장", cited_refs=["소득세법 제999조"])]

    async def fake_research_claim(claim, mode, settings, deadline):
        return [_make_evidence()]

    async def fake_verify_citations(refs):
        return {ref: False for ref in refs}  # 전부 미실재

    from app.rarr.agreement import AgreementResult

    async def fake_check_agreement(claim, evidence):
        return AgreementResult(agree=False, supporting=[])

    async def fake_edit_claim(claim, agreement, evidence):
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
