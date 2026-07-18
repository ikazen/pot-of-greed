from __future__ import annotations

import json
import time
import pytest

from app.rarr.types import Claim


@pytest.mark.asyncio
async def test_generate_questions_parses_json(monkeypatch):
    from app.rarr.query_gen import generate_questions

    questions = ["1세대1주택 비과세 요건은?", "보유기간 계산 기준일은?"]

    async def fake_chat(messages, *, system=None, json_mode=False, timeout=None):
        return json.dumps(questions)

    class FakeProvider:
        chat = staticmethod(fake_chat)

    import app.rarr.query_gen as qg_mod
    monkeypatch.setattr(qg_mod, "get_llm_provider", lambda role="default": FakeProvider())

    result = await generate_questions(Claim(text="1세대1주택 비과세는 보유기간 2년 이상이 필요하다."))
    assert result == questions


@pytest.mark.asyncio
async def test_generate_questions_fallback_on_llm_error(monkeypatch):
    from app.rarr.query_gen import generate_questions

    class ErrorProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            raise RuntimeError("LLM unavailable")

    import app.rarr.query_gen as qg_mod
    monkeypatch.setattr(qg_mod, "get_llm_provider", lambda role="default": ErrorProvider())

    claim = Claim(text="주장 텍스트")
    result = await generate_questions(claim)
    assert result == [claim.text]


# ---------------------------------------------------------------------------
# H1 — deadline 전파
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_questions_deadline_exceeded_skips_llm(monkeypatch):
    """deadline이 이미 지났으면 LLM 호출 없이 즉시 주장 텍스트로 폴백한다."""
    from app.rarr.query_gen import generate_questions
    calls = []

    class TrackingProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            calls.append(timeout)
            return json.dumps(["무시됨"])

    import app.rarr.query_gen as qg_mod
    monkeypatch.setattr(qg_mod, "get_llm_provider", lambda role="default": TrackingProvider())

    claim = Claim(text="주장 텍스트")
    result = await generate_questions(claim, deadline=time.monotonic() - 1)

    assert calls == []  # LLM 호출 안 됨
    assert result == [claim.text]


@pytest.mark.asyncio
async def test_generate_questions_deadline_clamps_timeout(monkeypatch):
    """deadline이 남아있으면 하드코딩 상한(10s)과 remaining 중 작은 값으로 timeout이 클램프된다."""
    from app.rarr.query_gen import generate_questions
    calls = []

    class TrackingProvider:
        async def chat(self, messages, *, system=None, json_mode=False, timeout=None):
            calls.append(timeout)
            return json.dumps(["질문"])

    import app.rarr.query_gen as qg_mod
    monkeypatch.setattr(qg_mod, "get_llm_provider", lambda role="default": TrackingProvider())

    claim = Claim(text="주장 텍스트")
    await generate_questions(claim, deadline=time.monotonic() + 3)

    assert len(calls) == 1
    assert calls[0] <= 3
