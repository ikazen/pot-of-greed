from __future__ import annotations

from typing import Literal

from app.agent.decompose import SubQuery

Route = Literal["hybrid", "graph"]


def route(subquery: SubQuery) -> Route:
    """하위질의 도구 힌트 기반 라우팅.

    tool_hint="graph"이거나 관계 탐색 키워드가 포함된 경우 graph 경로.
    그 외 hybrid 검색.
    """
    if subquery.tool_hint == "graph":
        return "graph"
    _GRAPH_KEYWORDS = ("준용", "인용", "관계", "개정", "변경", "OVERRULED", "AMENDED", "REFERS")
    if any(kw in subquery.text for kw in _GRAPH_KEYWORDS):
        return "graph"
    return "hybrid"
