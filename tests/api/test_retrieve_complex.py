from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_complex_mode_returns_chunks(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat", json={"query": "법인세법 제52조와 소득세법 제14조의 관계는?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "복잡 모드 RARR 답변입니다."
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_complex_mode_sources_dedup_contract(async_client, patch_retrieval, patch_rarr):
    """API 계약: sources는 chunk_id 기준 중복이 없어야 한다(pipeline 책임).

    실제 복잡 모드 검색 경로(구 _retrieve_complex)는 RARR(run_rarr)로 대체돼
    삭제됐다(#16) — 여기서는 /chat 엔드포인트가 pipeline 출력을 그대로
    전달하는 계약만 검증한다.
    """
    resp = await async_client.post("/chat", json={"query": "법인세법 제52조와 소득세법 제14조의 관계는?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    source_ids = [s["chunk_id"] for s in data["sources"]]
    assert len(source_ids) == len(set(source_ids))


@pytest.mark.asyncio
async def test_complex_mode_warnings_shape(async_client, patch_retrieval, patch_rarr):
    """API 계약: 200 + sources/warnings list."""
    resp = await async_client.post("/chat", json={"query": "법인세 부당행위계산?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["sources"], list)
    assert isinstance(data["warnings"], list)
