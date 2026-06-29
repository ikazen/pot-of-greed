from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.llm.ollama import OllamaProvider

_BASE = "http://localhost:11435"
_CHAT_URL = f"{_BASE}/api/chat"


@pytest.fixture
def provider():
    return OllamaProvider(base_url=_BASE, model="test-model")


@pytest.mark.asyncio
async def test_chat_basic(provider):
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=httpx.Response(
            200,
            json={"message": {"content": "답변입니다"}},
        ))
        result = await provider.chat(
            [{"role": "user", "content": "질의"}],
            system="시스템 프롬프트",
        )
    assert result == "답변입니다"


@pytest.mark.asyncio
async def test_chat_json_mode_sends_format_field(provider):
    captured = {}

    def capture(request, route):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"key": "val"}'}})

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=capture)
        await provider.chat(
            [{"role": "user", "content": "분해"}],
            system="sys",
            json_mode=True,
        )

    assert captured["body"].get("format") == "json"


@pytest.mark.asyncio
async def test_stream_chat_yields_tokens(provider):
    lines = [
        json.dumps({"message": {"content": "토큰1"}, "done": False}),
        json.dumps({"message": {"content": "토큰2"}, "done": True}),
    ]
    body = "\n".join(lines)

    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, text=body))
        tokens = []
        async for token in provider.stream_chat(
            [{"role": "user", "content": "질의"}],
            system="sys",
        ):
            tokens.append(token)

    assert tokens == ["토큰1", "토큰2"]


@pytest.mark.asyncio
async def test_chat_includes_api_key_header():
    provider = OllamaProvider(base_url=_BASE, model="m", api_key="secret")
    captured = {}

    def capture(request, route):
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=capture)
        await provider.chat([{"role": "user", "content": "q"}])

    assert captured["auth"] == "Bearer secret"
