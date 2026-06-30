from __future__ import annotations

import pytest

from app.rarr.agreement import AgreementResult
from app.rarr.types import Claim, Evidence


def _make_evidence(chunk_id="c1"):
    return Evidence(chunk_id=chunk_id, ref="소득세법 제89조", text="조문 내용", score=0.9, meta={})


def _agree(supporting=None):
    return AgreementResult(agree=True, supporting=supporting or [])


def _disagree():
    return AgreementResult(agree=False, supporting=[])


@pytest.mark.asyncio
async def test_edit_agree_claim_unchanged():
    from app.rarr.edit import edit_claim
    claim = Claim(text="정확한 주장")
    ev = _make_evidence()
    revised, used, corrections = await edit_claim(claim, _agree([ev]), [ev])
    assert revised == claim.text
    assert used == [ev]
    assert corrections == []


@pytest.mark.asyncio
async def test_edit_disagree_calls_llm(monkeypatch):
    expected = "수정된 주장 [정정: 틀린 번호 → 소득세법 제89조]"

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return expected

    import app.rarr.edit as edit_mod
    monkeypatch.setattr(edit_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.edit import edit_claim
    claim = Claim(text="잘못된 주장")
    ev = _make_evidence()
    revised, used, corrections = await edit_claim(claim, _disagree(), [ev])
    assert revised == expected
    assert "[정정:" in corrections[0]


@pytest.mark.asyncio
async def test_edit_no_evidence_adds_flag():
    from app.rarr.edit import edit_claim
    claim = Claim(text="근거 없는 주장")
    revised, used, corrections = await edit_claim(claim, _disagree(), [])
    assert "[미검증]" in revised
    assert used == []
    assert corrections == []


@pytest.mark.asyncio
async def test_edit_llm_failure_fallback(monkeypatch):
    class ErrorProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            raise RuntimeError("error")

    import app.rarr.edit as edit_mod
    monkeypatch.setattr(edit_mod, "get_llm_provider", lambda role="default": ErrorProvider())

    from app.rarr.edit import edit_claim
    claim = Claim(text="원문 주장")
    revised, used, corrections = await edit_claim(claim, _disagree(), [_make_evidence()])
    assert revised == claim.text
    assert used == []


def test_extract_corrections():
    from app.rarr.edit import _extract_corrections
    text = "수정된 주장 [정정: 틀린 번호 → 제89조] 추가 내용 [정정: 다른 수정]"
    corrections = _extract_corrections(text)
    assert len(corrections) == 2
    assert all("[정정:" in c for c in corrections)
