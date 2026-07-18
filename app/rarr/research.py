from __future__ import annotations

import asyncio
import time

from app.rarr.types import Claim, Evidence
from app.retrieval.vector_search import Chunk


def _chunk_to_evidence(chunk: Chunk) -> Evidence:
    if chunk.table == "article":
        ref = chunk.meta.get("law_name", "") + " " + chunk.meta.get("article_no", "")
    else:
        ref = chunk.meta.get("case_no", chunk.chunk_id)
    return Evidence(
        chunk_id=chunk.chunk_id,
        ref=ref.strip(),
        text=chunk.text,
        score=chunk.score,
        meta=chunk.meta,
    )


async def _research_simple(claim: Claim, settings) -> list[Chunk]:
    from app.api.chat import _retrieve_simple
    return await _retrieve_simple(claim.text, settings)


async def _research_complex(claim: Claim, settings, deadline: float) -> list[Chunk]:
    from app.api.chat import _search_complex
    from app.retrieval.reranker import rerank
    from app.retrieval.graph_expand import expand_2hop, filter_by_transaction_date
    from app.retrieval.context_expand import expand_to_parents
    from app.api.chat import _extract_transaction_date

    from app.rarr.query_gen import generate_questions

    questions = await generate_questions(claim, deadline=deadline)
    if settings.rarr_questions_per_claim:
        questions = questions[:settings.rarr_questions_per_claim]

    # M4: question마다 검색을 무제한 gather하면 claim 수 x question 수만큼 동시
    # DB 커넥션이 발사돼 asyncpg 풀을 고갈시킬 수 있다. semaphore로 동시 발사 수를 제한.
    semaphore = asyncio.Semaphore(settings.rarr_max_concurrency)

    async def _search_one(q: str) -> list[Chunk]:
        if time.monotonic() > deadline:
            return []
        async with semaphore:
            return await _search_complex(q, settings)

    results = await asyncio.gather(*[_search_one(q) for q in questions])

    merged: dict[str, Chunk] = {}
    for chunk_list in results:
        for c in chunk_list:
            if c.chunk_id not in merged or c.score > merged[c.chunk_id].score:
                merged[c.chunk_id] = c
    fused = sorted(merged.values(), key=lambda c: c.score, reverse=True)

    if time.monotonic() > deadline:
        return fused

    reranked = await rerank(claim.text, fused, top_k=settings.rerank_top_k)
    graph_chunks = await expand_2hop([c.chunk_id for c in reranked])

    txn_date = _extract_transaction_date(claim.text)
    if txn_date:
        graph_chunks = filter_by_transaction_date(graph_chunks, txn_date)

    graph_ids = {g.chunk_id for g in graph_chunks}
    reranked_ids = {c.chunk_id for c in reranked}
    extra = [c for c in fused if c.chunk_id in graph_ids and c.chunk_id not in reranked_ids]

    final = reranked + extra
    final += await expand_to_parents(final)
    return final


async def research_claim(
    claim: Claim,
    mode: str,
    settings,
    deadline: float,
) -> list[Evidence]:
    """주장 하나에 대해 코퍼스를 검색해 Evidence 목록을 반환.

    simple(RARR-lite): CQGen 생략, 주장 텍스트 직접 단일 검색.
    complex(full): CQGen + HyDE + 2hop + 시점필터.
    deadline 초과 시 조기 반환.
    """
    if time.monotonic() > deadline:
        return []

    if mode == "complex":
        chunks = await _research_complex(claim, settings, deadline)
    else:
        chunks = await _research_simple(claim, settings)

    return [_chunk_to_evidence(c) for c in chunks]
