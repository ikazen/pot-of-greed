from __future__ import annotations

import json
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
