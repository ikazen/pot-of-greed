from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.gemini import GeminiProvider


@pytest.fixture
def provider():
    return GeminiProvider(api_key="fake-key", model="gemini-2.5-flash")


def _make_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


@pytest.mark.asyncio
async def test_chat_returns_text(provider):
    mock_resp = _make_response("Gemini 답변")

    with patch.object(provider._client.aio.models, "generate_content", new=AsyncMock(return_value=mock_resp)):
        result = await provider.chat(
            [{"role": "user", "content": "소득세법 14조?"}],
            system="세법 전문가로 답하라",
        )

    assert result == "Gemini 답변"


@pytest.mark.asyncio
async def test_chat_json_mode_sets_response_mime_type(provider):
    captured_config = {}
    mock_resp = _make_response('{"sufficient": true}')

    async def fake_generate(model, contents, config):
        captured_config["cfg"] = config
        return mock_resp

    with patch.object(provider._client.aio.models, "generate_content", side_effect=fake_generate):
        await provider.chat(
            [{"role": "user", "content": "충분한가?"}],
            json_mode=True,
        )

    assert captured_config["cfg"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_stream_chat_yields_chunks(provider):
    chunks = [_make_response("청크1"), _make_response("청크2"), _make_response("")]

    async def fake_stream(*args, **kwargs):
        for c in chunks:
            yield c

    async def fake_generate_stream(model, contents, config):
        return fake_stream()

    with patch.object(provider._client.aio.models, "generate_content_stream", side_effect=fake_generate_stream):
        tokens = []
        async for token in provider.stream_chat(
            [{"role": "user", "content": "스트리밍"}],
            system="sys",
        ):
            tokens.append(token)

    assert tokens == ["청크1", "청크2"]


@pytest.mark.asyncio
async def test_chat_maps_assistant_role_to_model(provider):
    captured = {}
    mock_resp = _make_response("ok")

    async def fake_generate(model, contents, config):
        captured["contents"] = contents
        return mock_resp

    with patch.object(provider._client.aio.models, "generate_content", side_effect=fake_generate):
        await provider.chat([
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "네"},
            {"role": "user", "content": "계속"},
        ])

    roles = [c.role for c in captured["contents"]]
    assert roles == ["user", "model", "user"]
