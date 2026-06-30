from __future__ import annotations

import json
import re

from app.llm import get_llm_provider
from app.rarr.types import Claim

# 조문 번호: "XXX법 제N조" 형식
_RE_ARTICLE = re.compile(r"[\w가-힣]+법\s*제\d+조(?:\s*제\d+항)?")
# 판례 번호: 연도+사건부호+번호 (예: 2018두12345, 2021도1234)
_RE_CASE = re.compile(r"\d{4}[가-힣]+\d+")

_DECOMPOSE_SYSTEM = (
    "주어진 세법 답변을 독립적인(decontextualized) 원자 주장(atomic claim) 목록으로 분해하세요. "
    "각 주장은 다른 주장 없이도 의미가 통해야 합니다. "
    "JSON 배열만 반환하세요: [{\"text\": \"주장 내용\"}, ...]\n"
    "예시: [{\"text\": \"1세대1주택 비과세는 보유기간 2년 이상이 필요하다.\"}, ...]"
)


def _extract_refs(text: str) -> list[str]:
    refs = _RE_ARTICLE.findall(text) + _RE_CASE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for r in refs:
        r = r.strip()
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


async def decompose_claims(draft_text: str) -> list[Claim]:
    provider = get_llm_provider("aux")
    try:
        raw = await provider.chat(
            [{"role": "user", "content": draft_text}],
            system=_DECOMPOSE_SYSTEM,
            json_mode=True,
            timeout=15.0,
        )
        items = json.loads(raw.strip())
        claims = [
            Claim(text=item["text"], cited_refs=_extract_refs(item["text"]))
            for item in items
            if isinstance(item, dict) and item.get("text")
        ]
        if claims:
            return claims
    except Exception:
        pass
    return [Claim(text=draft_text, cited_refs=_extract_refs(draft_text))]
