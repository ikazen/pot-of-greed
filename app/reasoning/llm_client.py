from __future__ import annotations

from collections.abc import AsyncGenerator

from app.config import get_settings
from app.llm import get_llm_provider
from app.retrieval.vector_search import Chunk

_SIMPLE_SYSTEM = (
    "당신은 세법 전문 AI 어시스턴트입니다. "
    "주어진 세법 조문과 판례에 근거하여 정확하게 답변하십시오. "
    "답변의 모든 주장에 출처(조문번호 또는 판례번호)를 명시하십시오. "
    "근거가 없는 주장은 하지 마십시오."
)


def _build_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        if c.table == "article":
            ref = f"{c.meta.get('law_name', '')} {c.meta.get('article_no', '')} {c.meta.get('clause_path', '') or ''}".strip()
        else:
            ref = c.meta.get("case_no", c.chunk_id)
        parts.append(f"[{ref}]\n{c.text}")
    return "\n\n".join(parts)


async def simple_inference(query: str, chunks: list[Chunk]) -> str:
    context = _build_context(chunks)
    user_message = f"다음 법령/판례를 참고하여 질의에 답변하십시오.\n\n{context}\n\n질의: {query}"
    provider = get_llm_provider()
    return await provider.chat(
        [{"role": "user", "content": user_message}],
        system=_SIMPLE_SYSTEM,
    )


async def stream_simple_inference(
    query: str, chunks: list[Chunk]
) -> AsyncGenerator[str, None]:
    context = _build_context(chunks)
    user_message = f"다음 법령/판례를 참고하여 질의에 답변하십시오.\n\n{context}\n\n질의: {query}"
    provider = get_llm_provider()
    async for token in provider.stream_chat(
        [{"role": "user", "content": user_message}],
        system=_SIMPLE_SYSTEM,
    ):
        yield token


async def complex_inference(query: str, chunks: list[Chunk], system_extra: str = "") -> str:
    system = _SIMPLE_SYSTEM + ("\n" + system_extra if system_extra else "")
    settings = get_settings()
    context = _build_context(chunks)
    user_message = f"다음 법령/판례를 참고하여 질의에 상세히 답변하십시오.\n\n{context}\n\n질의: {query}"
    provider = get_llm_provider()
    return await provider.chat(
        [{"role": "user", "content": user_message}],
        system=system,
        timeout=float(settings.llm_timeout_s),
    )
