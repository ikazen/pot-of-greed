from __future__ import annotations

import pytest

from app.retrieval.vector_search import Chunk


@pytest.mark.asyncio
async def test_promotion_score_uses_raw_vector_top1(monkeypatch):
    """#11: 승격 판정은 벡터 top-1 코사인만 봐야 한다 — rerank/그래프 확장은 안 탄다."""
    chunk = Chunk("art_income_14", "article", "본문", 0.73,
                   {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True})

    async def fake_embed_query(text):
        return [0.1] * 1024

    calls = {"vector_search_top_k": None, "rerank_called": False, "expand_1hop_called": False}

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        calls["vector_search_top_k"] = top_k
        return [chunk]

    async def fake_rerank(query, chunks, top_k=None):
        calls["rerank_called"] = True
        return chunks

    async def fake_expand_1hop(chunk_ids):
        calls["expand_1hop_called"] = True
        return []

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)

    from app.api.chat import _promotion_score
    from app.config import get_settings

    score = await _promotion_score("소득세법 제14조?", get_settings())

    assert score == 0.73
    assert calls["vector_search_top_k"] == 1
    assert calls["rerank_called"] is False
    assert calls["expand_1hop_called"] is False


@pytest.mark.asyncio
async def test_promotion_score_empty_vector_results_returns_zero(monkeypatch):
    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return []

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)

    from app.api.chat import _promotion_score
    from app.config import get_settings

    score = await _promotion_score("애매한 질의", get_settings())
    assert score == 0.0


@pytest.mark.asyncio
async def test_retrieve_simple_hydrates_graph_only_chunk(monkeypatch):
    """#8: 검색 후보 풀(fused) 밖에서 1hop 그래프로만 발견된 chunk는
    드롭되지 않고 hydrate_by_ids로 본문이 채워져야 한다."""
    from app.retrieval.graph_expand import GraphChunk

    found_chunk = Chunk("art_income_14", "article", "소득세법 제14조 본문", 0.9,
                         {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True})
    graph_only_chunk = Chunk("art_graph_only", "article", "그래프 전용 조문 본문", 0.0,
                              {"law_name": "법인세법", "article_no": "제52조", "clause_path": None, "is_current": True})

    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return [found_chunk]

    async def fake_keyword_search(query, top_k=30):
        return [found_chunk]

    async def fake_rerank(query, chunks, top_k=None):
        return chunks[:top_k] if top_k else chunks

    async def fake_expand_1hop(chunk_ids):
        # fused/reranked 어디에도 없는 chunk_id를 반환 — hydration 필요 케이스
        return [GraphChunk(chunk_id="art_graph_only", node_type="article")]

    async def fake_expand_to_parents(chunks):
        return []

    hydrate_calls = []

    async def fake_hydrate_by_ids(chunk_ids):
        hydrate_calls.append(sorted(chunk_ids))
        return [graph_only_chunk]

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    monkeypatch.setattr("app.api.chat.hydrate_by_ids", fake_hydrate_by_ids)

    from app.api.chat import _retrieve_simple
    from app.config import get_settings

    result = await _retrieve_simple("법인세법 제52조와 소득세법 제14조", get_settings())

    assert hydrate_calls == [["art_graph_only"]]
    result_ids = {c.chunk_id for c in result}
    assert "art_graph_only" in result_ids
    hydrated = next(c for c in result if c.chunk_id == "art_graph_only")
    assert hydrated.text == "그래프 전용 조문 본문"


@pytest.mark.asyncio
async def test_retrieve_simple_no_hydration_when_graph_chunk_already_in_pool(monkeypatch):
    """그래프 확장 chunk가 이미 fused/reranked 안에 있으면 hydrate_by_ids를 호출하지 않는다."""
    from app.retrieval.graph_expand import GraphChunk

    found_chunk = Chunk("art_income_14", "article", "소득세법 제14조", 0.9,
                         {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True})

    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return [found_chunk]

    async def fake_keyword_search(query, top_k=30):
        return [found_chunk]

    async def fake_rerank(query, chunks, top_k=None):
        return chunks[:top_k] if top_k else chunks

    async def fake_expand_1hop(chunk_ids):
        return [GraphChunk(chunk_id="art_income_14", node_type="article")]  # 이미 reranked에 있음

    async def fake_expand_to_parents(chunks):
        return []

    hydrate_calls = []

    async def fake_hydrate_by_ids(chunk_ids):
        hydrate_calls.append(chunk_ids)
        return []

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    monkeypatch.setattr("app.api.chat.hydrate_by_ids", fake_hydrate_by_ids)

    from app.api.chat import _retrieve_simple
    from app.config import get_settings

    await _retrieve_simple("소득세법 제14조", get_settings())

    assert hydrate_calls == []
