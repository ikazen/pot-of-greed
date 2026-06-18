from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_simple_mode(async_client, patch_retrieval, patch_llm):
    resp = await async_client.post("/chat", json={"query": "부가가치세 신고 기한은?", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "단순 모드 답변입니다."
    assert "elapsed_ms" in data


@pytest.mark.asyncio
async def test_chat_complex_mode(async_client, patch_retrieval, patch_llm):
    resp = await async_client.post("/chat", json={"query": "법인세법 제52조 부당행위계산의 요건은?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "복잡 모드 답변입니다."


@pytest.mark.asyncio
async def test_chat_default_mode_is_simple(async_client, patch_retrieval, patch_llm):
    resp = await async_client.post("/chat", json={"query": "부가가치세 세율은?"})
    assert resp.status_code == 200
    # 기본값 simple, score=0.85 > 0.5이므로 단순 경로
    data = resp.json()
    assert data["answer"] == "단순 모드 답변입니다."


@pytest.mark.asyncio
async def test_chat_fallback_promotion(async_client, patch_low_score_retrieval, patch_llm):
    """top score=0.3 < threshold=0.5 → 자동 complex 승격."""
    resp = await async_client.post("/chat", json={"query": "애매한 질문", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    # fallback 승격으로 complex_inference가 호출됨
    assert data["answer"] == "복잡 모드 답변입니다."


@pytest.mark.asyncio
async def test_chat_sources_present(async_client, patch_retrieval, patch_llm):
    resp = await async_client.post("/chat", json={"query": "소득세법 과세표준?", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0


@pytest.mark.asyncio
async def test_chat_unauthorized():
    import httpx
    from app.main import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post("/chat", json={"query": "테스트"})
    assert resp.status_code == 401
