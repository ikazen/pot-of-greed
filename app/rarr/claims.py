from __future__ import annotations

import json
import re
import time

from app.llm import get_llm_provider
from app.rarr.types import Claim
from app.retrieval.refs import extract_refs

_DECOMPOSE_SYSTEM = (
    "주어진 세법 답변을 독립적인(decontextualized) 원자 주장(atomic claim) 목록으로 분해하세요. "
    "각 주장은 다른 주장 없이도 의미가 통해야 합니다. "
    "JSON 배열만 반환하세요: [{\"text\": \"주장 내용\"}, ...]\n"
    "예시: [{\"text\": \"1세대1주택 비과세는 보유기간 2년 이상이 필요하다.\"}, ...]"
)


_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """구두점/개행 기준 문장 분리 (rule-based, 폴백 전용).

    decompose LLM이 실패했을 때 draft 전체를 단일 claim으로 뭉치지 않기 위한
    최소한의 다항 유지책. 형태소 수준 분리는 하지 않는다.
    """
    parts = [p.strip() for p in _RE_SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


async def decompose_claims(draft_text: str, deadline: float | None = None) -> list[Claim]:
    """draft 텍스트를 원자 주장으로 분해.

    H1: deadline이 주어지면 남은 시간으로 timeout을 클램프하고, 이미 초과했으면
    LLM 호출 없이 규칙기반 폴백(_split_sentences)으로 직행한다.
    """
    timeout = 15.0
    skip_llm = False
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            skip_llm = True
        else:
            timeout = min(timeout, remaining)

    if not skip_llm:
        provider = get_llm_provider("aux")
        try:
            raw = await provider.chat(
                [{"role": "user", "content": draft_text}],
                system=_DECOMPOSE_SYSTEM,
                json_mode=True,
                timeout=timeout,
            )
            items = json.loads(raw.strip())
            claims = [
                Claim(text=item["text"], cited_refs=extract_refs(item["text"]))
                for item in items
                if isinstance(item, dict) and item.get("text")
            ]
            if claims:
                return claims
        except Exception:
            pass

    sentences = _split_sentences(draft_text)
    if not sentences:
        return [Claim(text=draft_text, cited_refs=extract_refs(draft_text))]
    return [Claim(text=s, cited_refs=extract_refs(s)) for s in sentences]
