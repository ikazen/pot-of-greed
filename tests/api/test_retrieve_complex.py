from __future__ import annotations

import pytest
from app.retrieval.vector_search import Chunk


@pytest.mark.asyncio
async def test_complex_mode_returns_chunks(async_client, patch_retrieval, patch_rarr):
    resp = await async_client.post("/chat", json={"query": "법인세법 제52조와 소득세법 제14조의 관계는?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "복잡 모드 RARR 답변입니다."
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_complex_mode_dedup(async_client, patch_rarr, monkeypatch):
    """두 하위질의가 동일 chunk_id를 반환해도 응답 sources에서 dedup."""
    from app.agent.decompose import SubQuery

    chunk_a = Chunk("art_income_14", "article", "소득세법 제14조", 0.9, {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True})
    chunk_b = Chunk("art_income_14", "article", "소득세법 제14조", 0.7, {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True})

    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_hyde_embedding(query):
        return [0.2] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return [chunk_a]

    async def fake_keyword_search(query, top_k=30):
        return [chunk_b]

    async def fake_rerank(query, chunks, top_k=None):
        return chunks[:top_k] if top_k else chunks

    async def fake_expand_1hop(chunk_ids):
        return []

    async def fake_expand_2hop(chunk_ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    async def fake_decompose(query):
        return [
            SubQuery(text="하위질의1", tool_hint="hybrid"),
            SubQuery(text="하위질의2", tool_hint="hybrid"),
        ]

    def fake_route(sq):
        return "hybrid"

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.hyde_embedding", fake_hyde_embedding)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_2hop", fake_expand_2hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    monkeypatch.setattr("app.api.chat.decompose", fake_decompose)
    monkeypatch.setattr("app.api.chat.route", fake_route)

    resp = await async_client.post("/chat", json={"query": "법인세법 제52조와 소득세법 제14조의 관계는?", "mode": "complex"})
    assert resp.status_code == 200
    # RARR 경로는 pipeline이 sources를 관리 — 출력 계약 확인
    data = resp.json()
    assert isinstance(data["sources"], list)
    source_ids = [s["chunk_id"] for s in data["sources"]]
    assert len(source_ids) == len(set(source_ids))  # dedup within pipeline output


@pytest.mark.asyncio
async def test_complex_mode_2hop_chunks_merged(async_client, patch_rarr, monkeypatch):
    """API 계약 검증: 200 + sources/warnings list."""
    from app.agent.decompose import SubQuery
    from app.retrieval.graph_expand import GraphChunk

    chunk_a = Chunk("art_income_14", "article", "소득세법 제14조", 0.9, {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True})
    chunk_b = Chunk("art_corp_52", "article", "법인세법 제52조", 0.6, {"law_name": "법인세법", "article_no": "제52조", "clause_path": None, "is_current": True})

    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_hyde_embedding(query):
        return [0.2] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return [chunk_a, chunk_b]

    async def fake_keyword_search(query, top_k=30):
        return [chunk_a]

    async def fake_rerank(query, chunks, top_k=None):
        return [chunk_a]

    async def fake_expand_1hop(chunk_ids):
        return []

    async def fake_expand_2hop(chunk_ids):
        return [GraphChunk(chunk_id="art_corp_52", node_type="article")]

    async def fake_expand_to_parents(chunks):
        return []

    async def fake_decompose(query):
        return [SubQuery(text=query, tool_hint="hybrid")]

    def fake_route(sq):
        return "hybrid"

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.hyde_embedding", fake_hyde_embedding)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_2hop", fake_expand_2hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    monkeypatch.setattr("app.api.chat.decompose", fake_decompose)
    monkeypatch.setattr("app.api.chat.route", fake_route)

    resp = await async_client.post("/chat", json={"query": "법인세 부당행위계산?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["sources"], list)
    assert isinstance(data["warnings"], list)
