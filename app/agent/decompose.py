from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

ToolHint = Literal["hybrid", "graph"]


@dataclass
class SubQuery:
    text: str
    tool_hint: ToolHint = "hybrid"


_DECOMPOSE_SYSTEM = (
    "당신은 세법 질의 분석 어시스턴트입니다. "
    "복합 질의를 독립적인 쟁점별 하위질의로 분해하십시오. "
    "각 하위질의에 적합한 도구 힌트를 지정하십시오: "
    "'hybrid'(일반 검색), 'graph'(판례 간 관계 또는 법령 준용관계 탐색 필요). "
    "반드시 JSON 배열만 출력하고 설명은 금지: "
    '[{"text": "하위질의", "tool_hint": "hybrid|graph"}, ...]'
)


async def decompose(query: str) -> list[SubQuery]:
    """복합 세법 질의를 쟁점별 하위질의로 분해.

    LLM 호출 실패 또는 JSON 파싱 오류 시 전체 쿼리를 단일 hybrid 하위질의로 반환.
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
                        {"role": "system", "content": _DECOMPOSE_SYSTEM},
                        {"role": "user", "content": f"다음 질의를 분해하십시오: {query}"},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        items = json.loads(raw)
        subqueries = [
            SubQuery(
                text=item["text"],
                tool_hint=item.get("tool_hint", "hybrid") if item.get("tool_hint") in ("hybrid", "graph") else "hybrid",
            )
            for item in items
            if item.get("text")
        ]
        if subqueries:
            return subqueries
    except Exception:
        logger.debug("decompose fallback to single query", exc_info=True)
    return [SubQuery(text=query, tool_hint="hybrid")]
