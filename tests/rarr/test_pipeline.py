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
