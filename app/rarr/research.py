from __future__ import annotations

import asyncio
import time

from app.rarr.types import Claim, Evidence
from app.retrieval.pipeline import retrieve_simple, search_complex
from app.retrieval.types import Chunk
from app.retrieval.vector_search import hydrate_by_ids


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
    return await retrieve_simple(claim.text, settings)


async def _research_complex(
    claim: Claim,
    settings,
    deadline: float,
    search_semaphore: asyncio.Semaphore | None = None,
) -> list[Chunk]:
    from app.retrieval.reranker import rerank
    from app.retrieval.graph_expand import expand_2hop, filter_by_transaction_date
    from app.retrieval.context_expand import expand_to_parents
    # app.api.chat이 app.rarr.pipeline(→ research)을 임포트하므로, 이 방향의
    # 참조는 최상단에 두면 순환 임포트가 된다 — 함수 호출 시점 지연 임포트로 유지.
    from app.api.chat import _extract_transaction_date

    from app.rarr.query_gen import generate_questions

    questions = await generate_questions(claim, deadline=deadline)
    if settings.rarr_questions_per_claim:
        questions = questions[:settings.rarr_questions_per_claim]

    # #15: 이 함수는 claim마다 호출되므로, 매번 새 세마포어를 만들면 claim 동시성(N)
    # x 이 세마포어(N)가 중첩돼 최악 N^2개 서브쿼리 검색이 동시 발사된다(M4 원 버그).
    # run_rarr가 만든 하나의 search_semaphore를 claim들 사이에서 공유해 "전체 동시
    # 서브쿼리 검색 수"를 rarr_max_concurrency로 단일 상한한다. 직접 호출(테스트 등)
    # 시엔 None 폴백으로 이 함수 단독 동작도 유지.
    semaphore = search_semaphore or asyncio.Semaphore(settings.rarr_max_concurrency)

    async def _search_one(q: str) -> list[Chunk]:
        if time.monotonic() > deadline:
            return []
        async with semaphore:
            return await search_complex(q, settings)

    results = await asyncio.gather(*[_search_one(q) for q in questions])

    merged: dict[str, Chunk] = {}
    for chunk_list in results:
        for c in chunk_list:
            if c.chunk_id not in merged or c.score > merged[c.chunk_id].score:
                merged[c.chunk_id] = c
    fused = sorted(merged.values(), key=lambda c: c.score, reverse=True)

    if time.monotonic() > deadline:
        return fused

    reranked = await rerank(claim.text, fused, settings)
    graph_chunks = await expand_2hop([c.chunk_id for c in reranked])

    txn_date = _extract_transaction_date(claim.text)
    if txn_date:
        graph_chunks = filter_by_transaction_date(graph_chunks, txn_date)

    graph_ids = {g.chunk_id for g in graph_chunks}
    reranked_ids = {c.chunk_id for c in reranked}
    fused_ids = {c.chunk_id for c in fused}
    extra = [c for c in fused if c.chunk_id in graph_ids and c.chunk_id not in reranked_ids]
    # #8: fused/reranked 밖에서 그래프로만 발견된 chunk는 PG에서 본문을 직접 채운다.
    missing_ids = graph_ids - fused_ids - reranked_ids
    hydrated = await hydrate_by_ids(list(missing_ids)) if missing_ids else []

    final = reranked + extra + hydrated
    final += await expand_to_parents(final)
    return final


async def research_claim(
    claim: Claim,
    mode: str,
    settings,
    deadline: float,
    search_semaphore: asyncio.Semaphore | None = None,
) -> list[Evidence]:
    """주장 하나에 대해 코퍼스를 검색해 Evidence 목록을 반환.

    simple(RARR-lite): CQGen 생략, 주장 텍스트 직접 단일 검색.
    complex(full): CQGen + HyDE + 2hop + 시점필터.
    deadline 초과 시 조기 반환.
    """
    if time.monotonic() > deadline:
        return []

    if mode == "complex":
        chunks = await _research_complex(claim, settings, deadline, search_semaphore)
    else:
        chunks = await _research_simple(claim, settings)

    return [_chunk_to_evidence(c) for c in chunks]
