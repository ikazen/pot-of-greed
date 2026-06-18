from __future__ import annotations

import json
import pytest
import respx
import httpx

from app.agent.decompose import decompose, SubQuery


@pytest.fixture
def ollama_url():
    from app.config import get_settings
    return f"{get_settings().ollama_cloud_base_url}/api/chat"


@pytest.mark.asyncio
async def test_decompose_multi_subquery(ollama_url):
    payload = [
        {"text": "법인세법 제52조 부당행위계산의 요건", "tool_hint": "hybrid"},
        {"text": "소득세법 제14조와의 관계", "tool_hint": "graph"},
    ]
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200,
            json={"message": {"content": json.dumps(payload)}},
        ))
        result = await decompose("법인세법 제52조와 소득세법 제14조의 관계는?")

    assert len(result) == 2
    assert result[0].text == "법인세법 제52조 부당행위계산의 요건"
    assert result[0].tool_hint == "hybrid"
    assert result[1].tool_hint == "graph"


@pytest.mark.asyncio
async def test_decompose_fallback_on_json_error(ollama_url):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200,
            json={"message": {"content": "JSON 아님"}},
        ))
        result = await decompose("부가가치세 면세 대상은?")

    assert len(result) == 1
    assert result[0].text == "부가가치세 면세 대상은?"
    assert result[0].tool_hint == "hybrid"


@pytest.mark.asyncio
async def test_decompose_fallback_on_http_error(ollama_url):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(500))
        result = await decompose("세금 신고 기한은?")

    assert len(result) == 1
    assert result[0].text == "세금 신고 기한은?"


@pytest.mark.asyncio
async def test_decompose_invalid_tool_hint_normalized(ollama_url):
    payload = [{"text": "부가가치세 세율", "tool_hint": "unknown_hint"}]
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200,
            json={"message": {"content": json.dumps(payload)}},
        ))
        result = await decompose("부가가치세?")

    assert result[0].tool_hint == "hybrid"
