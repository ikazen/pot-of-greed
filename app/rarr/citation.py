from __future__ import annotations

import asyncio

from app.retrieval.keyword_search import keyword_search


async def verify_citations(refs: list[str]) -> dict[str, bool]:
    """조문/판례 번호 목록을 tsvector 정확매칭으로 코퍼스 존재 확인.

    LLM·의미검색 없음. 결과 없으면 할루시네이션 플래그.
    반환: {ref: 존재여부}
    """
    if not refs:
        return {}

    async def _check(ref: str) -> tuple[str, bool]:
        results = await keyword_search(ref, top_k=1)
        return ref, bool(results)

    pairs = await asyncio.gather(*[_check(ref) for ref in refs])
    return dict(pairs)
