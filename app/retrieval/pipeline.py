from __future__ import annotations

import asyncio

from app.agent.decompose import decompose
from app.retrieval.context_expand import expand_to_parents
from app.retrieval.embedder import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.graph_expand import expand_1hop
from app.retrieval.hyde import hyde_embedding
from app.retrieval.keyword_search import keyword_search
from app.retrieval.reranker import rerank
from app.retrieval.types import Chunk
from app.retrieval.vector_search import hydrate_by_ids, vector_search


async def promotion_score(query: str, settings) -> float:
    """승격 판정 전용 — 벡터 top-1 코사인만 확인(#11).

    임베딩+2검색+리랭크+그래프+parent확장까지 도는 전체 retrieve_simple을
    승격 신호 하나 읽으려고 돌리고 버리던 낭비를 제거한다. retrieve_simple
    자체는 app.rarr.research(claim별 실제 검색)에서 계속 쓰인다.
    """
    embedding = await embed_query(query)
    vec_chunks = await vector_search(embedding, top_k=1)
    return vec_chunks[0].score if vec_chunks else 0.0


async def parallel_search(
    embedding: list[float],
    query: str,
    top_k: int,
) -> tuple[list[Chunk], list[Chunk]]:
    vec_task = asyncio.create_task(vector_search(embedding, top_k=top_k))
    kw_task = asyncio.create_task(keyword_search(query, top_k=top_k))
    vec_chunks, kw_chunks = await asyncio.gather(vec_task, kw_task)
    return vec_chunks, kw_chunks


async def retrieve_simple(query: str, settings) -> list[Chunk]:
    embedding = await embed_query(query)
    vec_chunks, kw_chunks = await parallel_search(embedding, query, settings.retrieve_top_k)
    fused = rrf_fuse(vec_chunks, kw_chunks, k=settings.rrf_k, top_n=settings.retrieve_top_k)
    reranked = await rerank(query, fused, top_k=settings.rerank_top_k)
    extra_graph = await expand_1hop([c.chunk_id for c in reranked])
    extra_chunk_ids = {g.chunk_id for g in extra_graph}
    reranked_ids = {r.chunk_id for r in reranked}
    fused_ids = {c.chunk_id for c in fused}
    in_pool = [c for c in fused if c.chunk_id in extra_chunk_ids and c.chunk_id not in reranked_ids]
    # #8: 검색 후보 풀(fused) 밖에서 그래프로만 발견된 chunk는 본문을 PG에서 직접 채운다
    # — 그렇지 않으면 id만 알고 텍스트가 없어 통째로 드롭된다.
    missing_ids = extra_chunk_ids - fused_ids - reranked_ids
    hydrated = await hydrate_by_ids(list(missing_ids)) if missing_ids else []
    final_chunks = reranked + in_pool + hydrated
    final_chunks += await expand_to_parents(final_chunks)
    return final_chunks


async def search_complex(query: str, settings) -> list[Chunk]:
    """§5.2 1~4단계 단일 패스: 분해 → 도구 라우팅 → HyDE + 하이브리드 검색 → Neo4j 2홉."""
    subqueries = await decompose(query)

    async def _search_subquery(sq) -> list[Chunk]:
        # tool_hint 기반 그래프 전용 라우팅(app.agent.tool_router.route)은 아직 그래프
        # 전용 검색 백엔드가 없어 미배선 상태 — 향후 연결 지점(#16).
        direct_emb, hyde_emb = await asyncio.gather(
            embed_query(sq.text),
            hyde_embedding(sq.text),
        )
        vec_direct, vec_hyde, kw = await asyncio.gather(
            vector_search(direct_emb, top_k=settings.retrieve_top_k),
            vector_search(hyde_emb, top_k=settings.retrieve_top_k),
            keyword_search(sq.text, top_k=settings.retrieve_top_k),
        )
        return rrf_fuse(
            vec_direct,
            vec_hyde,
            kw,
            k=settings.rrf_k,
            top_n=settings.retrieve_top_k,
        )

    results = await asyncio.gather(*[_search_subquery(sq) for sq in subqueries])

    merged: dict[str, Chunk] = {}
    for chunk_list in results:
        for c in chunk_list:
            if c.chunk_id not in merged or c.score > merged[c.chunk_id].score:
                merged[c.chunk_id] = c
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)
