from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lawcorpus.search import promotion_score

from app.auth.jwt import get_current_user
from app.config import get_settings
from app.router.mode_classifier import classify, should_promote
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
    debug: dict | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    _: str = Depends(get_current_user),
) -> ChatResponse:
    t0 = time.monotonic()
    settings = get_settings()
    mode = classify(req.query, req.mode)

    # RARR 강도 노브: simple=RARR-lite, complex=full RARR (결정 M)
    top_score = await promotion_score(req.query, settings)
    if should_promote(top_score, settings.fallback_score_threshold):
        mode = "complex"

    result = await run_rarr(req.query, mode, settings)
    elapsed = int((time.monotonic() - t0) * 1000)
    return ChatResponse(
        answer=result.answer,
        sources=[vars(s) for s in result.sources],
        warnings=[vars(w) for w in result.warnings],
        elapsed_ms=elapsed,
        debug=result.debug,
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    _: str = Depends(get_current_user),
) -> StreamingResponse:
    settings = get_settings()
    mode = classify(req.query, req.mode)

    top_score = await promotion_score(req.query, settings)
    if should_promote(top_score, settings.fallback_score_threshold):
        mode = "complex"

    async def _event_stream():
        yield f"data: {json.dumps({'status': '검토 중'})}\n\n"

        # #13: run_rarr을 백그라운드 task로 돌리고, on_progress 콜백이 큐에 쌓는
        # 진행상태(draft 완료/분해 완료/검증 n/총)를 폴링해 중간에 흘린다 —
        # 이전엔 "검토 중" 한 줄 후 완료까지 침묵하는 가짜 스트리밍이었다.
        queue: asyncio.Queue[str] = asyncio.Queue()

        def on_progress(status: str) -> None:
            queue.put_nowait(status)

        task = asyncio.create_task(
            run_rarr(req.query, mode, settings, on_progress=on_progress)
        )
        while not task.done():
            try:
                status = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'status': status})}\n\n"
            except asyncio.TimeoutError:
                continue
        while not queue.empty():
            yield f"data: {json.dumps({'status': queue.get_nowait()})}\n\n"

        result = task.result()

        # 최종 답변 청크 단위 스트리밍
        chunk_size = 20
        answer = result.answer
        for i in range(0, len(answer), chunk_size):
            yield f"data: {json.dumps({'token': answer[i:i + chunk_size]})}\n\n"

        tail = {
            "sources": [vars(s) for s in result.sources],
            "warnings": [vars(w) for w in result.warnings],
        }
        if result.debug is not None:
            tail["debug"] = result.debug
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


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
