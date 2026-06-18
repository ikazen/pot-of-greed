from __future__ import annotations

import httpx

from app.config import get_settings
from app.retrieval.embedder import embed_query

_HYDE_SYSTEM = (
    "당신은 세법 전문가입니다. "
    "다음 세법 질의에 대한 가상의 법령 조문 또는 판례 요지를 간결하게 작성하십시오. "
    "실제 근거가 없어도 됩니다. 검색용 가상 문서 생성이 목적입니다."
)


async def hyde_embedding(query: str) -> list[float]:
    """HyDE: 가상 답변 문서 생성 → 임베딩.

    LLM 호출 실패 시 쿼리 직접 임베딩으로 폴백.
    """
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.ollama_cloud_base_url}/api/chat",
                headers={"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else {},
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {"role": "system", "content": _HYDE_SYSTEM},
                        {"role": "user", "content": query},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
        hypothetical = resp.json()["message"]["content"].strip()
    except Exception:
        hypothetical = query
    return await embed_query(hypothetical)
