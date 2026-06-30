from __future__ import annotations

import json
import pytest

from app.rarr.types import Claim, Evidence


def _make_evidence(chunk_id="c1", ref="소득세법 제89조"):
    return Evidence(chunk_id=chunk_id, ref=ref, text="관련 조문 내용", score=0.9, meta={})


@pytest.mark.asyncio
async def test_check_agreement_agree(monkeypatch):
    response = json.dumps({"agree": True, "supporting_ids": ["c1"], "reason": "일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="소득세법 제89조에 따라 비과세된다.")
    ev = _make_evidence("c1")
    result = await check_agreement(claim, [ev])
    assert result.agree is True
    assert result.supporting == [ev]


@pytest.mark.asyncio
async def test_check_agreement_disagree(monkeypatch):
    response = json.dumps({"agree": False, "supporting_ids": [], "reason": "불일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="잘못된 주장")
    result = await check_agreement(claim, [_make_evidence()])
    assert result.agree is False
    assert result.supporting == []


@pytest.mark.asyncio
async def test_check_agreement_no_evidence():
    from app.rarr.agreement import check_agreement
    claim = Claim(text="주장")
    result = await check_agreement(claim, [])
    assert result.agree is False


@pytest.mark.asyncio
async def test_check_agreement_llm_failure(monkeypatch):
    class ErrorProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            raise RuntimeError("error")

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": ErrorProvider())

    from app.rarr.agreement import check_agreement
    result = await check_agreement(Claim(text="주장"), [_make_evidence()])
    assert result.agree is False
