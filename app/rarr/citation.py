from __future__ import annotations

from app.db.pg import get_pool
from app.rarr.claims import parse_ref


async def verify_citations(refs: list[str]) -> dict[str, bool]:
    """조문/판례 번호 목록을 구조적 동등 매칭으로 코퍼스 존재 확인.

    LLM·의미검색·tsvector 랭킹 없음 — law_name+article_no 또는 case_no의
    정확한 컬럼 동등성만 본다. FTS 토큰 AND 매칭은 "법명은 틀리고 번호만
    실재하는" 오귀속 인용을 통과시키므로 사용하지 않는다.
    반환: {ref: 존재여부}. 구조 파싱 불가 ref는 보수적으로 False.
    """
    if not refs:
        return {}

    pool = get_pool()
    out: dict[str, bool] = {}
    async with pool.acquire() as conn:
        for ref in refs:
            parsed = parse_ref(ref)
            if parsed is None:
                out[ref] = False
                continue
            kind, params = parsed
            if kind == "article":
                row = await conn.fetchval(
                    "SELECT 1 FROM article_chunks WHERE law_name = $1 AND article_no = $2 LIMIT 1",
                    *params,
                )
            else:
                row = await conn.fetchval(
                    "SELECT 1 FROM case_chunks WHERE case_no = $1 LIMIT 1",
                    *params,
                )
            out[ref] = row is not None
    return out
