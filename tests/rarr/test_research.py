from __future__ import annotations

import time
import pytest

from app.rarr.types import Claim, Evidence


def _make_chunk(chunk_id="c1", table="article", score=0.9):
    from app.retrieval.vector_search import Chunk
    meta = {"law_name": "소득세법", "article_no": "제89조"} if table == "article" else {"case_no": "2018두123"}
    return Chunk(chunk_id=chunk_id, table=table, text="sample text", score=score, meta=meta)


@pytest.mark.asyncio
async def test_research_claim_simple_returns_evidence(monkeypatch):
    chunk = _make_chunk()

    async def fake_retrieve_simple(query, settings):
        return [chunk]

    import app.rarr.research as research_mod
    monkeypatch.setattr(research_mod, "_research_simple", fake_retrieve_simple)

    from app.rarr.research import research_claim
    claim = Claim(text="1세대1주택 비과세 요건")
    result = await research_claim(claim, "simple", object(), deadline=time.monotonic() + 10)

    assert len(result) == 1
    assert isinstance(result[0], Evidence)
    assert result[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_research_claim_simple_article_ref_format(monkeypatch):
    chunk = _make_chunk(table="article")

    async def fake_retrieve_simple(query, settings):
        return [chunk]

    import app.rarr.research as research_mod
    monkeypatch.setattr(research_mod, "_research_simple", fake_retrieve_simple)

    from app.rarr.research import research_claim
    claim = Claim(text="소득세법 제89조")
    result = await research_claim(claim, "simple", object(), deadline=time.monotonic() + 10)
    assert result[0].ref == "소득세법 제89조"


@pytest.mark.asyncio
async def test_research_claim_deadline_exceeded_returns_empty(monkeypatch):
    async def fake_retrieve_simple(query, settings):
        return [_make_chunk()]

    import app.rarr.research as research_mod
    monkeypatch.setattr(research_mod, "_research_simple", fake_retrieve_simple)

    from app.rarr.research import research_claim
    claim = Claim(text="test")
    # deadline already passed
    result = await research_claim(claim, "simple", object(), deadline=time.monotonic() - 1)
    assert result == []


@pytest.mark.asyncio
async def test_research_claim_complex_calls_complex_path(monkeypatch):
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    called = []

    async def fake_research_complex(claim, settings, deadline):
        called.append(True)
        return chunks

    import app.rarr.research as research_mod
    monkeypatch.setattr(research_mod, "_research_complex", fake_research_complex)

    from app.rarr.research import research_claim
    claim = Claim(text="복잡 질의")
    result = await research_claim(claim, "complex", object(), deadline=time.monotonic() + 10)
    assert called
    assert len(result) == 2
