from __future__ import annotations

import asyncio

from app.agent.decompose import decompose
from app.rag.hyde import hyde_embedding
from lawcorpus.retrieval.embedder import embed_query
from lawcorpus.retrieval.fusion import rrf_fuse
from lawcorpus.retrieval.keyword_search import keyword_search
from lawcorpus.retrieval.vector_search import vector_search
from lawcorpus.types import Chunk


async def search_complex(query: str, settings) -> list[Chunk]:
    """§5.2 1~4단계 단일 패스: 분해 → 도구 라우팅 → HyDE + 하이브리드 검색 → Neo4j 2홉.

    질의분해(decompose)와 HyDE 둘 다 LLM 호출이라 the-book-of-moon 라이브러리에 넣지 않고
    여기(app/rag/)에 둔다 — the-book-of-moon 결정 A(LLM 비의존 원칙) 참조.
    """
    subqueries = await decompose(query)

    async def _search_subquery(sq) -> list[Chunk]:
        # tool_hint 기반 그래프 전용 라우팅(app.agent.tool_router.route)은 아직 그래프
        # 전용 검색 백엔드가 없어 미배선 상태 — 향후 연결 지점(#16).
        direct_emb, hyde_emb = await asyncio.gather(
            embed_query(sq.text, settings),
            hyde_embedding(sq.text, settings),
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
