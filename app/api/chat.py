from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.jwt import get_current_user
from app.config import get_settings
from app.retrieval.embedder import embed_query
from app.retrieval.vector_search import vector_search
from app.retrieval.keyword_search import keyword_search
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import rerank
from app.retrieval.graph_expand import expand_1hop
from app.retrieval.vector_search import Chunk
from app.reasoning.llm_client import simple_inference
from app.reasoning.answer_builder import build_answer

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

    embedding = await embed_query(req.query)

    vec_chunks, kw_chunks = await _parallel_search(embedding, req.query, settings.retrieve_top_k)

    fused = rrf_fuse(vec_chunks, kw_chunks, k=settings.rrf_k, top_n=settings.retrieve_top_k)

    reranked = await rerank(req.query, fused, top_k=settings.rerank_top_k)

    extra_graph = await expand_1hop([c.chunk_id for c in reranked])
    extra_chunk_ids = {g.chunk_id for g in extra_graph}
    final_chunks = reranked + [
        c for c in fused if c.chunk_id in extra_chunk_ids and c.chunk_id not in {r.chunk_id for r in reranked}
    ]

    raw_answer = await simple_inference(req.query, final_chunks)

    answer = build_answer(raw_answer, final_chunks)
    elapsed = int((time.monotonic() - t0) * 1000)

    return ChatResponse(
        answer=answer.answer,
        sources=[vars(s) for s in answer.sources],
        warnings=[vars(w) for w in answer.warnings],
        elapsed_ms=elapsed,
    )


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
