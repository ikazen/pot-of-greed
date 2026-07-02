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


# ---------------------------------------------------------------------------
# C4 — 인용-근거 정합 불변식 (cited_refs ⊆ supporting)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_agreement_cited_ref_in_supporting(monkeypatch):
    response = json.dumps({"agree": True, "supporting_ids": ["c1"], "reason": "일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="소득세법 제89조에 따라 비과세된다.", cited_refs=["소득세법 제89조"])
    ev = _make_evidence("c1", ref="소득세법 제89조")
    result = await check_agreement(claim, [ev])
    assert result.agree is True


@pytest.mark.asyncio
async def test_check_agreement_cited_ref_not_in_supporting_downgrades(monkeypatch):
    """LLM이 agree=True를 줘도, 인용한 ref가 supporting 근거에 없으면(무관한 실재 인용)
    agree를 False로 강등해 edit 경로에 회부해야 한다."""
    response = json.dumps({"agree": True, "supporting_ids": ["c1"], "reason": "일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="소득세법 제89조에 따라 비과세된다.", cited_refs=["법인세법 제1조"])
    ev = _make_evidence("c1", ref="소득세법 제89조")
    result = await check_agreement(claim, [ev])
    assert result.agree is False


@pytest.mark.asyncio
async def test_check_agreement_clause_ref_normalized_to_article(monkeypatch):
    """cited_refs의 항(제N항) 표기는 조 단위로 정규화되어 supporting과 매칭돼야 한다."""
    response = json.dumps({"agree": True, "supporting_ids": ["c1"], "reason": "일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="소득세법 제14조 제1항에 따라 과세표준을 계산한다.", cited_refs=["소득세법 제14조 제1항"])
    ev = _make_evidence("c1", ref="소득세법 제14조")
    result = await check_agreement(claim, [ev])
    assert result.agree is True


@pytest.mark.asyncio
async def test_check_agreement_no_cited_refs_unaffected(monkeypatch):
    """cited_refs가 없는 claim은 불변식의 영향을 받지 않는다(회귀 확인)."""
    response = json.dumps({"agree": True, "supporting_ids": ["c1"], "reason": "일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="근거에 부합하는 일반적인 주장")
    ev = _make_evidence("c1", ref="소득세법 제89조")
    result = await check_agreement(claim, [ev])
    assert result.agree is True


@pytest.mark.asyncio
async def test_check_agreement_cited_case_no_not_in_supporting_downgrades(monkeypatch):
    response = json.dumps({"agree": True, "supporting_ids": ["c1"], "reason": "일치"})

    class FakeProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return response

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": FakeProvider())

    from app.rarr.agreement import check_agreement
    claim = Claim(text="대법원 판례에 따라 과세된다.", cited_refs=["2018두12345"])
    ev = _make_evidence("c1", ref="소득세법 제89조")
    result = await check_agreement(claim, [ev])
    assert result.agree is False


# ---------------------------------------------------------------------------
# H1 — deadline 전파
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_agreement_deadline_exceeded_skips_llm(monkeypatch):
    """deadline이 이미 지났으면 LLM 호출 없이 즉시 degrade한다."""
    calls = []

    class TrackingProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            calls.append(timeout)
            return json.dumps({"agree": True, "supporting_ids": [], "reason": ""})

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": TrackingProvider())

    import time
    from app.rarr.agreement import check_agreement
    result = await check_agreement(Claim(text="주장"), [_make_evidence()], deadline=time.monotonic() - 1)

    assert result.agree is False
    assert calls == []  # LLM 호출 안 됨


@pytest.mark.asyncio
async def test_check_agreement_deadline_clamps_timeout(monkeypatch):
    """deadline이 남아있으면 하드코딩 상한(15s)과 remaining 중 작은 값으로 timeout이 클램프된다."""
    calls = []

    class TrackingProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            calls.append(timeout)
            return json.dumps({"agree": True, "supporting_ids": [], "reason": ""})

    import app.rarr.agreement as agreement_mod
    monkeypatch.setattr(agreement_mod, "get_llm_provider", lambda role="default": TrackingProvider())

    import time
    from app.rarr.agreement import check_agreement
    await check_agreement(Claim(text="주장"), [_make_evidence()], deadline=time.monotonic() + 3)

    assert len(calls) == 1
    assert calls[0] <= 3


class TestCitationsGrounded:
    def test_empty_cited_refs(self):
        from app.rarr.agreement import _citations_grounded
        assert _citations_grounded([], [_make_evidence(ref="소득세법 제89조")]) is True

    def test_subset(self):
        from app.rarr.agreement import _citations_grounded
        ev = _make_evidence(ref="소득세법 제89조")
        assert _citations_grounded(["소득세법 제89조"], [ev]) is True

    def test_not_subset(self):
        from app.rarr.agreement import _citations_grounded
        ev = _make_evidence(ref="소득세법 제89조")
        assert _citations_grounded(["법인세법 제1조"], [ev]) is False

    def test_unparseable_cited_ref_excluded(self):
        from app.rarr.agreement import _citations_grounded
        ev = _make_evidence(ref="소득세법 제89조")
        assert _citations_grounded(["아무말"], [ev]) is True
