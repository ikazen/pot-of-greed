from __future__ import annotations

import pytest
import respx
import httpx


@pytest.fixture
def ollama_url():
    from app.config import get_settings
    return f"{get_settings().ollama_cloud_base_url}/api/chat"


@pytest.fixture
def ollama_embed_url():
    from app.config import get_settings
    return f"{get_settings().ollama_base_url}/api/embed"


@pytest.mark.asyncio
async def test_hyde_uses_hypothetical_text(ollama_url, ollama_embed_url):
    from app.retrieval.hyde import hyde_embedding

    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200,
            json={"message": {"content": "소득세법 제14조에 의하면 과세표준은..."}},
        ))
        respx.post(ollama_embed_url).mock(return_value=httpx.Response(
            200,
            json={"embeddings": [[0.5] * 1024]},
        ))
        result = await hyde_embedding("소득세 과세표준 계산 방법은?")

    assert len(result) == 1024
    assert result[0] == 0.5


@pytest.mark.asyncio
async def test_hyde_fallback_on_llm_error(ollama_url, ollama_embed_url):
    from app.retrieval.hyde import hyde_embedding

    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(500))
        respx.post(ollama_embed_url).mock(return_value=httpx.Response(
            200,
            json={"embeddings": [[0.1] * 1024]},
        ))
        result = await hyde_embedding("세금 신고 기한")

    assert len(result) == 1024
