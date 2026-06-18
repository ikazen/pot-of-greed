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
from app.retrieval.graph_expand import expand_1hop
from app.retrieval.context_expand import expand_to_parents
from app.retrieval.vector_search import Chunk
from app.reasoning.llm_client import simple_inference, complex_inference, stream_simple_inference
from app.reasoning.answer_builder import build_answer
from app.router.mode_classifier import classify, should_promote

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

    simple_chunks = await _retrieve_simple(req.query, settings)
    top_score = simple_chunks[0].score if simple_chunks else 0.0

    if mode == "complex" or should_promote(top_score, settings.fallback_score_threshold):
        final_chunks = await _retrieve_complex(req.query, settings, simple_chunks)
        raw_answer = await complex_inference(req.query, final_chunks)
    else:
        final_chunks = simple_chunks
        raw_answer = await simple_inference(req.query, final_chunks)

    answer = build_answer(raw_answer, final_chunks)
    elapsed = int((time.monotonic() - t0) * 1000)
    return ChatResponse(
        answer=answer.answer,
        sources=[vars(s) for s in answer.sources],
        warnings=[vars(w) for w in answer.warnings],
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

    if mode == "complex" or should_promote(top_score, settings.fallback_score_threshold):
        final_chunks = await _retrieve_complex(req.query, settings, simple_chunks)
    else:
        final_chunks = simple_chunks

    async def _event_stream():
        collected: list[str] = []
        async for token in stream_simple_inference(req.query, final_chunks):
            collected.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        raw_answer = "".join(collected)
        answer = build_answer(raw_answer, final_chunks)
        tail = {
            "sources": [vars(s) for s in answer.sources],
            "warnings": [vars(w) for w in answer.warnings],
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


async def _retrieve_complex(query: str, settings, simple_chunks: list[Chunk]) -> list[Chunk]:
    # BON-144에서 분해→라우팅→HyDE→2홉으로 채워진다.
    # 이 단계에선 단순 검색 결과를 그대로 사용 (골격만 확보).
    return simple_chunks


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
