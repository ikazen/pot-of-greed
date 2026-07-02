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


@pytest.mark.asyncio
async def test_research_complex_questions_per_claim_cap(monkeypatch):
    """rarr_questions_per_claim=2이면 질문 5개 생성해도 검색이 2번만 호출된다."""
    import app.rarr.research as research_mod
    from app.rarr.types import Claim

    search_calls = []

    async def fake_generate_questions(claim):
        return ["q1", "q2", "q3", "q4", "q5"]

    async def fake_search_complex(query, settings):
        search_calls.append(query)
        return []

    async def fake_rerank(query, chunks, top_k):
        return []

    async def fake_expand_2hop(ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    # lazy imports inside _research_complex read from source module objects
    import app.rarr.query_gen as qg_mod
    import app.api.chat as chat_mod
    from app.retrieval import reranker as reranker_mod
    from app.retrieval import graph_expand as ge_mod
    from app.retrieval import context_expand as ce_mod

    monkeypatch.setattr(qg_mod, "generate_questions", fake_generate_questions)
    monkeypatch.setattr(chat_mod, "_search_complex", fake_search_complex)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)
    monkeypatch.setattr(ge_mod, "expand_2hop", fake_expand_2hop)
    monkeypatch.setattr(ce_mod, "expand_to_parents", fake_expand_to_parents)

    class FakeSettings:
        rarr_questions_per_claim = 2
        rerank_top_k = 5
        rarr_max_concurrency = 4

    claim = Claim(text="복잡 질의")
    await research_mod._research_complex(claim, FakeSettings(), deadline=time.monotonic() + 30)
    assert len(search_calls) == 2
