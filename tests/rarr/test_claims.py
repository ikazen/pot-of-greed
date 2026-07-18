from __future__ import annotations

import json
import time
import pytest


@pytest.fixture
def mock_aux_llm(monkeypatch):
    """aux role LLM을 monkeypatch로 대체."""
    from app.llm import get_llm_provider
    get_llm_provider.cache_clear()

    class FakeProvider:
        def __init__(self, response: str):
            self._response = response

        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            return self._response

    return FakeProvider


def test_extract_refs_from_article_text():
    from app.rarr.claims import _extract_refs
    text = "소득세법 제89조에 따르면 1세대1주택은 비과세된다."
    refs = _extract_refs(text)
    assert "소득세법 제89조" in refs


def test_extract_refs_from_case_text():
    from app.rarr.claims import _extract_refs
    text = "대법원 2018두12345 판결에서 이 기준을 확립했다."
    refs = _extract_refs(text)
    assert "2018두12345" in refs


def test_extract_refs_empty_when_no_refs():
    from app.rarr.claims import _extract_refs
    assert _extract_refs("보유기간 2년 이상이 필요하다.") == []


def test_extract_refs_ignores_date_expressions():
    """#6: 날짜 표현("2018년6월15일")이 판례번호로 오탐되어선 안 된다."""
    from app.rarr.claims import _extract_refs
    text = "양도일이 2018년6월15일인 경우 소득세법 제89조가 적용된다."
    refs = _extract_refs(text)
    assert "소득세법 제89조" in refs
    assert "2018년6" not in refs
    assert not any(r.startswith("2018년") for r in refs)


def test_extract_refs_various_case_type_codes():
    """사건부호 화이트리스트가 실무에서 흔한 부호들을 계속 인식하는지 확인."""
    from app.rarr.claims import _extract_refs
    text = "2015다12345, 2020도6789, 2019헌가1 판결을 참고하라."
    refs = _extract_refs(text)
    assert "2015다12345" in refs
    assert "2020도6789" in refs
    assert "2019헌가1" in refs


@pytest.mark.asyncio
async def test_decompose_claims_parses_json(monkeypatch):
    from app.rarr.claims import decompose_claims

    items = [
        {"text": "1세대1주택 비과세는 보유기간 2년 이상이 필요하다."},
        {"text": "소득세법 제89조에서 규정한다."},
    ]

    async def fake_chat(messages, *, system=None, json_mode=False, timeout=None):
        return json.dumps(items)

    class FakeProvider:
        chat = staticmethod(fake_chat)

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": FakeProvider())

    result = await decompose_claims("소득세법 제89조에 따른 1세대1주택 비과세 요건은?")
    assert len(result) == 2
    assert result[0].text == items[0]["text"]
    assert "소득세법 제89조" in result[1].cited_refs


@pytest.mark.asyncio
async def test_decompose_claims_fallback_on_llm_error(monkeypatch):
    from app.rarr.claims import decompose_claims

    class ErrorProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            raise RuntimeError("LLM unavailable")

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": ErrorProvider())

    draft = "1세대1주택 비과세 요건은 보유기간 2년이다."
    result = await decompose_claims(draft)
    assert len(result) == 1
    assert result[0].text == draft


@pytest.mark.asyncio
async def test_decompose_claims_fallback_on_empty_list(monkeypatch):
    from app.rarr.claims import decompose_claims

    async def fake_chat(messages, *, system=None, json_mode=False, timeout=None):
        return "[]"

    class FakeProvider:
        chat = staticmethod(fake_chat)

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": FakeProvider())

    draft = "단일 주장 초안."
    result = await decompose_claims(draft)
    assert len(result) == 1
    assert result[0].text == draft


@pytest.mark.asyncio
async def test_decompose_claims_fallback_splits_multi_sentence_draft(monkeypatch):
    from app.rarr.claims import decompose_claims

    class ErrorProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            raise RuntimeError("LLM unavailable")

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": ErrorProvider())

    draft = "1세대1주택 비과세는 보유기간 2년이 필요하다. 소득세법 제89조에서 규정한다."
    result = await decompose_claims(draft)

    assert len(result) == 2
    assert result[0].text == "1세대1주택 비과세는 보유기간 2년이 필요하다."
    assert result[1].text == "소득세법 제89조에서 규정한다."
    assert result[0].cited_refs == []
    assert "소득세법 제89조" in result[1].cited_refs


@pytest.mark.asyncio
async def test_cited_refs_extracted_from_decomposed_claims(monkeypatch):
    from app.rarr.claims import decompose_claims

    items = [{"text": "소득세법 제89조 제1항에 해당한다."}]

    async def fake_chat(messages, *, system=None, json_mode=False, timeout=None):
        return json.dumps(items)

    class FakeProvider:
        chat = staticmethod(fake_chat)

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": FakeProvider())

    result = await decompose_claims("any")
    assert result[0].cited_refs  # refs 추출됨


# ---------------------------------------------------------------------------
# H1 — deadline 전파
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decompose_claims_deadline_exceeded_skips_llm(monkeypatch):
    """deadline이 이미 지났으면 LLM 호출 없이 규칙기반 폴백으로 직행한다."""
    calls = []

    class TrackingProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            calls.append(timeout)
            return json.dumps([{"text": "무시됨"}])

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": TrackingProvider())

    from app.rarr.claims import decompose_claims
    draft = "1세대1주택 비과세는 보유기간 2년이 필요하다."
    result = await decompose_claims(draft, deadline=time.monotonic() - 1)

    assert calls == []  # LLM 호출 안 됨
    assert result[0].text == draft


@pytest.mark.asyncio
async def test_decompose_claims_deadline_clamps_timeout(monkeypatch):
    """deadline이 남아있으면 하드코딩 상한(15s)과 remaining 중 작은 값으로 timeout이 클램프된다."""
    calls = []

    class TrackingProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            calls.append(timeout)
            return json.dumps([{"text": "주장"}])

    import app.rarr.claims as claims_mod
    monkeypatch.setattr(claims_mod, "get_llm_provider", lambda role="default": TrackingProvider())

    from app.rarr.claims import decompose_claims
    await decompose_claims("초안", deadline=time.monotonic() + 3)

    assert len(calls) == 1
    assert calls[0] <= 3
