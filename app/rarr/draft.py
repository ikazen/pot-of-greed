from __future__ import annotations

from app.llm import get_llm_provider

_DRAFT_SYSTEM = (
    "당신은 세법 전문가입니다. 질문에 대해 정확하고 상세하게 답변하세요. "
    "조문 번호(예: 소득세법 제89조)나 판례 번호(예: 2018두12345)를 알고 있다면 명시하세요. "
    "검색 결과 없이 알고 있는 지식으로만 답변합니다."
)


async def draft(query: str, timeout: float | None = None) -> str:
    provider = get_llm_provider("draft")
    return await provider.chat(
        [{"role": "user", "content": query}],
        system=_DRAFT_SYSTEM,
        timeout=timeout,
    )
