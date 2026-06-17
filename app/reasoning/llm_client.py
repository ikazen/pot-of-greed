from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx

from app.config import get_settings
from app.retrieval.vector_search import Chunk

# 단순 모드 시스템 프롬프트
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
    """단순 모드 단일 추론 패스."""
    settings = get_settings()
    context = _build_context(chunks)
    user_message = f"다음 법령/판례를 참고하여 질의에 답변하십시오.\n\n{context}\n\n질의: {query}"

    async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
        resp = await client.post(
            f"{settings.ollama_cloud_base_url}/api/chat",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else {},
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": _SIMPLE_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


async def stream_simple_inference(
    query: str, chunks: list[Chunk]
) -> AsyncGenerator[str, None]:
    """단순 모드 스트리밍 추론 — 토큰을 yield."""
    settings = get_settings()
    context = _build_context(chunks)
    user_message = f"다음 법령/판례를 참고하여 질의에 답변하십시오.\n\n{context}\n\n질의: {query}"

    async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_cloud_base_url}/api/chat",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else {},
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": _SIMPLE_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk_data = json.loads(line)
                token = chunk_data.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk_data.get("done"):
                    break


async def complex_inference(query: str, chunks: list[Chunk], system_extra: str = "") -> str:
    """복잡 모드 추론 seam — BON-142~147에서 확장 진입점."""
    system = _SIMPLE_SYSTEM + ("\n" + system_extra if system_extra else "")
    settings = get_settings()
    context = _build_context(chunks)
    user_message = f"다음 법령/판례를 참고하여 질의에 상세히 답변하십시오.\n\n{context}\n\n질의: {query}"

    async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
        resp = await client.post(
            f"{settings.ollama_cloud_base_url}/api/chat",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else {},
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]
