from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.config import get_settings
from app.llm import get_llm_provider
from app.retrieval.vector_search import Chunk

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], Awaitable[list[Chunk]]]


@dataclass
class SufficiencyResult:
    sufficient: bool
    rewritten_query: str | None = None


_SUFFICIENCY_SYSTEM = (
    "당신은 세법 검색 품질 평가자입니다. "
    "제공된 검색 결과가 사용자 질의를 답변하기에 충분한지 판단하십시오. "
    "충분하면 JSON: {\"sufficient\": true} "
    "부족하면 JSON: {\"sufficient\": false, \"rewritten_query\": \"보완된 검색 쿼리\"} "
    "다른 텍스트는 출력하지 마십시오."
)


async def evaluate(query: str, chunks: list[Chunk]) -> SufficiencyResult:
    """현재 검색 결과로 질의를 답변하기에 충분한지 LLM으로 판단.

    파싱 실패 또는 LLM 오류 시 sufficient=True로 폴백 (루프 종료).
    """
    context_preview = "\n".join(
        f"[{c.chunk_id}] {c.text[:200]}" for c in chunks[:5]
    )
    user_msg = f"질의: {query}\n\n검색 결과:\n{context_preview}"

    try:
        provider = get_llm_provider()
        raw = await provider.chat(
            [{"role": "user", "content": user_msg}],
            system=_SUFFICIENCY_SYSTEM,
            json_mode=True,
            timeout=8.0,
        )
        data = json.loads(raw.strip())
        return SufficiencyResult(
            sufficient=bool(data.get("sufficient", True)),
            rewritten_query=data.get("rewritten_query"),
        )
    except Exception:
        logger.debug("sufficiency evaluate fallback to sufficient=True", exc_info=True)
        return SufficiencyResult(sufficient=True)


async def sufficiency_loop(
    query: str,
    retrieve_fn: RetrieveFn,
    settings,
    deadline: float,
) -> list[Chunk]:
    """충분성 루프 — 최대 sufficiency_max_iter 반복, deadline 추적 조기 탈출.

    deadline: time.monotonic() 기반 절대 시각 (complex_mode_timeout_s 기준).
    """
    max_iter = min(settings.sufficiency_max_iter, 3)
    chunks = await retrieve_fn(query)

    for _ in range(max_iter):
        if time.monotonic() >= deadline:
            logger.debug("sufficiency_loop: deadline exceeded, early exit")
            break

        result = await evaluate(query, chunks)
        if result.sufficient:
            break

        if not result.rewritten_query:
            break

        if time.monotonic() >= deadline:
            logger.debug("sufficiency_loop: deadline exceeded after evaluate")
            break

        query = result.rewritten_query
        chunks = await retrieve_fn(query)

    return chunks
