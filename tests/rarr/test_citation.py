from __future__ import annotations

import pytest

from app.rarr.types import Claim


@pytest.mark.asyncio
async def test_verify_citations_existing(monkeypatch):
    from app.retrieval.vector_search import Chunk

    async def fake_keyword_search(query: str, top_k: int = 30):
        return [Chunk(chunk_id="c1", table="article", text="...", score=0.9, meta={})]

    import app.rarr.citation as citation_mod
    monkeypatch.setattr(citation_mod, "keyword_search", fake_keyword_search)

    from app.rarr.citation import verify_citations
    result = await verify_citations(["소득세법 제89조"])
    assert result["소득세법 제89조"] is True


@pytest.mark.asyncio
async def test_verify_citations_hallucinated(monkeypatch):
    async def fake_keyword_search(query: str, top_k: int = 30):
        return []

    import app.rarr.citation as citation_mod
    monkeypatch.setattr(citation_mod, "keyword_search", fake_keyword_search)

    from app.rarr.citation import verify_citations
    result = await verify_citations(["존재하지않는법 제9999조"])
    assert result["존재하지않는법 제9999조"] is False


@pytest.mark.asyncio
async def test_verify_citations_empty_refs():
    from app.rarr.citation import verify_citations
    result = await verify_citations([])
    assert result == {}


@pytest.mark.asyncio
async def test_verify_citations_multiple(monkeypatch):
    from app.retrieval.vector_search import Chunk

    async def fake_keyword_search(query: str, top_k: int = 30):
        if "소득세법" in query:
            return [Chunk(chunk_id="c1", table="article", text="...", score=0.9, meta={})]
        return []

    import app.rarr.citation as citation_mod
    monkeypatch.setattr(citation_mod, "keyword_search", fake_keyword_search)

    from app.rarr.citation import verify_citations
    result = await verify_citations(["소득세법 제89조", "없는법 제1조"])
    assert result["소득세법 제89조"] is True
    assert result["없는법 제1조"] is False
