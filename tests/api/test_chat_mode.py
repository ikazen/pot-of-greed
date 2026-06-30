from __future__ import annotations

import json
import pytest


@pytest.mark.asyncio
async def test_chat_simple_mode(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat", json={"query": "부가가치세 신고 기한은?", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "단순 모드 RARR 답변입니다."
    assert "elapsed_ms" in data


@pytest.mark.asyncio
async def test_chat_complex_mode(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat", json={"query": "법인세법 제52조 부당행위계산의 요건은?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "복잡 모드 RARR 답변입니다."


@pytest.mark.asyncio
async def test_chat_default_mode_is_simple(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat", json={"query": "부가가치세 세율은?"})
    assert resp.status_code == 200
    # 기본값 simple, score=0.85 > 0.5이므로 단순 경로
    data = resp.json()
    assert data["answer"] == "단순 모드 RARR 답변입니다."


@pytest.mark.asyncio
async def test_chat_fallback_promotion(async_client, patch_low_score_retrieval, patch_rarr):
    """top score=0.3 < threshold=0.5 → should_promote → RARR complex."""
    resp = await async_client.post("/chat", json={"query": "애매한 질문", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "복잡 모드 RARR 답변입니다."


@pytest.mark.asyncio
async def test_chat_sources_and_warnings_shape(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat", json={"query": "소득세법 과세표준?", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    source = data["sources"][0]
    # 출력 계약: {type, ref, chunk_id, summary}
    assert {"type", "ref", "chunk_id", "summary"} <= source.keys()
    assert isinstance(data["warnings"], list)


@pytest.mark.asyncio
async def test_chat_safety_net_degrade(async_client, patch_retrieval, monkeypatch):
    """RARR 실패 시 [미검증] 배너 degrade."""
    from app.rarr.pipeline import RarrResult

    async def failing_run_rarr(query, mode, settings):
        return RarrResult(
            answer="초안 답변입니다.\n\n[미검증] 답변 검증에 실패했습니다. 내용을 반드시 확인하세요.",
            sources=[],
            warnings=[],
            attributions=[],
        )

    monkeypatch.setattr("app.api.chat.run_rarr", failing_run_rarr)
    resp = await async_client.post("/chat", json={"query": "테스트", "mode": "simple"})
    assert resp.status_code == 200
    data = resp.json()
    assert "[미검증]" in data["answer"]
    assert data["sources"] == []


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


@pytest.mark.asyncio
async def test_chat_stream_returns_sse(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat/stream", json={"query": "부가가치세?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "data:" in body
    assert "[DONE]" in body


@pytest.mark.asyncio
async def test_chat_stream_contains_tokens_and_tail(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat/stream", json={"query": "부가가치세?"})
    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data:")]
    # status, 토큰들, tail, [DONE]
    payloads = []
    for ln in lines:
        raw = ln[len("data:"):].strip()
        if raw == "[DONE]":
            continue
        payloads.append(json.loads(raw))

    status_frames = [p for p in payloads if "status" in p]
    token_frames = [p for p in payloads if "token" in p]
    tail_frames = [p for p in payloads if "sources" in p]

    assert status_frames  # "검토 중" 상태 프레임
    assert token_frames   # 답변 토큰
    assert tail_frames    # sources/warnings
    assert "sources" in tail_frames[-1]
    assert "warnings" in tail_frames[-1]
