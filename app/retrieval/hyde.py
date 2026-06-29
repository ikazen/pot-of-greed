from __future__ import annotations

from app.llm import get_llm_provider
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
    try:
        provider = get_llm_provider()
        hypothetical = await provider.chat(
            [{"role": "user", "content": query}],
            system=_HYDE_SYSTEM,
            timeout=10.0,
        )
        hypothetical = hypothetical.strip() or query
    except Exception:
        hypothetical = query
    return await embed_query(hypothetical)
