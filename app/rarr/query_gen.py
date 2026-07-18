from __future__ import annotations

import json
import time

from app.llm import get_llm_provider
from app.rarr.types import Claim

_QGEN_SYSTEM = (
    "주어진 세법 주장을 검증하기 위한 검색 질문 목록을 생성하세요. "
    "각 질문은 해당 주장의 법적 근거를 코퍼스에서 찾을 수 있도록 구체적이어야 합니다. "
    "JSON 배열만 반환하세요: [\"질문1\", \"질문2\", ...]"
)


async def generate_questions(claim: Claim, deadline: float | None = None) -> list[str]:
    """주장 검증을 위한 검색 질문 생성 (complex 모드 전용). 폴백: 주장 텍스트 1개.

    H1: deadline이 주어지면 남은 시간으로 timeout을 클램프하고, 이미 초과했으면
    LLM 호출 없이 즉시 폴백한다.
    """
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return [claim.text]
        timeout = min(10.0, remaining)
    else:
        timeout = 10.0

    provider = get_llm_provider("aux")
    try:
        raw = await provider.chat(
            [{"role": "user", "content": claim.text}],
            system=_QGEN_SYSTEM,
            json_mode=True,
            timeout=timeout,
        )
        items = json.loads(raw.strip())
        questions = [q for q in items if isinstance(q, str) and q.strip()]
        if questions:
            return questions
    except Exception:
        pass
    return [claim.text]
