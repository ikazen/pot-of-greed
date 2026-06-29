from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from app.llm.ollama import OllamaProvider
from app.llm.gemini import GeminiProvider

_BASE = "http://localhost:11435"
_CHAT_URL = f"{_BASE}/api/chat"


@pytest.mark.asyncio
async def test_ollama_on_request_called_with_payload():
    captured = []
    provider = OllamaProvider(base_url=_BASE, model="m", on_request=captured.append)

    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=httpx.Response(
            200, json={"message": {"content": "ok"}}
        ))
        await provider.chat(
            [{"role": "user", "content": "질의"}],
            system="sys",
        )

    assert len(captured) == 1
    req = captured[0]
    assert req["transport"] == f"POST {_BASE}/api/chat"
    assert req["body"]["model"] == "m"
    assert req["body"]["stream"] is False
    # system 메시지가 첫 번째
    assert req["body"]["messages"][0]["role"] == "system"
    assert req["body"]["messages"][0]["content"] == "sys"


@pytest.mark.asyncio
async def test_ollama_on_request_masks_api_key():
    captured = []
    provider = OllamaProvider(base_url=_BASE, model="m", api_key="secret", on_request=captured.append)

    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=httpx.Response(
            200, json={"message": {"content": "ok"}}
        ))
        await provider.chat([{"role": "user", "content": "q"}])

    req = captured[0]
    assert req["headers"].get("Authorization") == "***"


@pytest.mark.asyncio
async def test_ollama_stream_on_request_called():
    captured = []
    provider = OllamaProvider(base_url=_BASE, model="m", on_request=captured.append)

    lines = [
        json.dumps({"message": {"content": "t"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, text="\n".join(lines)))
        async for _ in provider.stream_chat([{"role": "user", "content": "q"}]):
            pass

    assert len(captured) == 1
    assert captured[0]["body"]["stream"] is True


@pytest.mark.asyncio
async def test_gemini_on_request_called_with_payload():
    captured = []
    provider = GeminiProvider(api_key="fake", model="gemini-test", on_request=captured.append)

    mock_resp = MagicMock()
    mock_resp.text = "응답"

    with patch.object(provider._client.aio.models, "generate_content", new=AsyncMock(return_value=mock_resp)):
        await provider.chat(
            [{"role": "user", "content": "질의"}],
            system="sys",
        )

    assert len(captured) == 1
    req = captured[0]
    assert "generateContent" in req["transport"]
    assert req["model"] == "gemini-test"
    assert isinstance(req["contents"], list)
    assert req["contents"][0]["parts"][0]["text"] == "질의"
    assert req["config"]["system_instruction"] == "sys"


@pytest.mark.asyncio
async def test_gemini_json_mode_in_hook():
    captured = []
    provider = GeminiProvider(api_key="fake", model="m", on_request=captured.append)
    mock_resp = MagicMock()
    mock_resp.text = '{"a": 1}'

    with patch.object(provider._client.aio.models, "generate_content", new=AsyncMock(return_value=mock_resp)):
        await provider.chat([{"role": "user", "content": "q"}], json_mode=True)

    assert captured[0]["config"]["response_mime_type"] == "application/json"


@pytest.mark.asyncio
async def test_make_llm_provider_wires_hook(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", _BASE)
    from app.config import get_settings
    from app.llm import make_llm_provider
    get_settings.cache_clear()

    captured = []
    provider = make_llm_provider(on_request=captured.append)

    assert isinstance(provider, OllamaProvider)

    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=httpx.Response(
            200, json={"message": {"content": "ok"}}
        ))
        await provider.chat([{"role": "user", "content": "q"}])

    assert len(captured) == 1
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_make_llm_provider_model_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", _BASE)
    from app.config import get_settings
    from app.llm import make_llm_provider
    get_settings.cache_clear()

    provider = make_llm_provider(model="custom-model")
    assert isinstance(provider, OllamaProvider)
    assert provider._model == "custom-model"

    get_settings.cache_clear()
