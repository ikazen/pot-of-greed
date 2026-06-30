from __future__ import annotations

import json
import time
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.jwt import get_current_user
from app.config import get_settings
from app.retrieval.embedder import embed_query
from app.retrieval.vector_search import vector_search
from app.retrieval.keyword_search import keyword_search
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import rerank
from app.retrieval.graph_expand import expand_1hop, expand_2hop, filter_by_transaction_date
from app.retrieval.context_expand import expand_to_parents
from app.retrieval.hyde import hyde_embedding
from app.retrieval.vector_search import Chunk
from app.router.mode_classifier import classify, should_promote
from app.agent.decompose import decompose
from app.agent.tool_router import route
from app.agent.sufficiency import sufficiency_loop
from app.agent.grounding_check import check_answer, apply_grounding
from app.rarr.pipeline import run_rarr

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    mode: Literal["simple", "complex"] = "simple"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    warnings: list[dict]
    elapsed_ms: int


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    _: str = Depends(get_current_user),
) -> ChatResponse:
    t0 = time.monotonic()
    settings = get_settings()
    mode = classify(req.query, req.mode)

    # RARR 강도 노브: simple=RARR-lite, complex=full RARR (결정 M)
    simple_chunks = await _retrieve_simple(req.query, settings)
    top_score = simple_chunks[0].score if simple_chunks else 0.0
    if should_promote(top_score, settings.fallback_score_threshold):
        mode = "complex"

    result = await run_rarr(req.query, mode, settings)
    elapsed = int((time.monotonic() - t0) * 1000)
    return ChatResponse(
        answer=result.answer,
        sources=[vars(s) for s in result.sources],
        warnings=[vars(w) for w in result.warnings],
        elapsed_ms=elapsed,
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    _: str = Depends(get_current_user),
) -> StreamingResponse:
    settings = get_settings()
    mode = classify(req.query, req.mode)

    simple_chunks = await _retrieve_simple(req.query, settings)
    top_score = simple_chunks[0].score if simple_chunks else 0.0
    if should_promote(top_score, settings.fallback_score_threshold):
        mode = "complex"

    async def _event_stream():
        yield f"data: {json.dumps({'status': '검토 중'})}\n\n"

        result = await run_rarr(req.query, mode, settings)

        # 최종 답변 청크 단위 스트리밍
        chunk_size = 20
        answer = result.answer
        for i in range(0, len(answer), chunk_size):
            yield f"data: {json.dumps({'token': answer[i:i + chunk_size]})}\n\n"

        tail = {
            "sources": [vars(s) for s in result.sources],
            "warnings": [vars(w) for w in result.warnings],
        }
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


async def _retrieve_simple(query: str, settings) -> list[Chunk]:
    embedding = await embed_query(query)
    vec_chunks, kw_chunks = await _parallel_search(embedding, query, settings.retrieve_top_k)
    fused = rrf_fuse(vec_chunks, kw_chunks, k=settings.rrf_k, top_n=settings.retrieve_top_k)
    reranked = await rerank(query, fused, top_k=settings.rerank_top_k)
    extra_graph = await expand_1hop([c.chunk_id for c in reranked])
    extra_chunk_ids = {g.chunk_id for g in extra_graph}
    final_chunks = reranked + [
        c for c in fused if c.chunk_id in extra_chunk_ids and c.chunk_id not in {r.chunk_id for r in reranked}
    ]
    final_chunks += await expand_to_parents(final_chunks)
    return final_chunks


async def _search_complex(query: str, settings) -> list[Chunk]:
    """§5.2 1~4단계 단일 패스: 분해 → 도구 라우팅 → HyDE + 하이브리드 검색 → Neo4j 2홉."""
    import asyncio

    subqueries = await decompose(query)

    async def _search_subquery(sq) -> list[Chunk]:
        route(sq)  # 라우팅 결정 (graph 경로는 향후 Neo4j 직접 탐색으로 확장)
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
            vec_direct + vec_hyde,
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


def _extract_transaction_date(query: str) -> str | None:
    """질의에서 거래시점 날짜 추출 (ISO 형식 반환). 미발견 시 None."""
    import re
    # ISO: 2018-06-01 / 2018.06.01
    m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", query)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # 한국어: 2018년 6월 15일
    m = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", query)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # 연월만: 2021년 3월
    m = re.search(r"(\d{4})년\s*(\d{1,2})월", query)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-01"
    return None


async def _retrieve_complex(query: str, settings, simple_chunks: list[Chunk]) -> list[Chunk]:
    """§5.2 1~5단계: 분해/HyDE/2홉 + 충분성 루프 + 2층 시점 필터."""
    deadline = time.monotonic() + settings.complex_mode_timeout_s

    fused_all = await sufficiency_loop(
        query,
        lambda q: _search_complex(q, settings),
        settings,
        deadline,
    )

    reranked = await rerank(query, fused_all, top_k=settings.rerank_top_k)

    graph_chunks = await expand_2hop([c.chunk_id for c in reranked])

    # 2층 시점 정합: 질의에 거래시점 명시 시 조문 유효범위 필터
    txn_date = _extract_transaction_date(query)
    if txn_date:
        graph_chunks = filter_by_transaction_date(graph_chunks, txn_date)

    graph_ids = {g.chunk_id for g in graph_chunks}
    reranked_ids = {c.chunk_id for c in reranked}
    extra = [c for c in fused_all if c.chunk_id in graph_ids and c.chunk_id not in reranked_ids]

    final_chunks = reranked + extra
    final_chunks += await expand_to_parents(final_chunks)
    return final_chunks


async def _parallel_search(
    embedding: list[float],
    query: str,
    top_k: int,
) -> tuple[list[Chunk], list[Chunk]]:
    import asyncio
    vec_task = asyncio.create_task(vector_search(embedding, top_k=top_k))
    kw_task = asyncio.create_task(keyword_search(query, top_k=top_k))
    vec_chunks, kw_chunks = await asyncio.gather(vec_task, kw_task)
    return vec_chunks, kw_chunks
