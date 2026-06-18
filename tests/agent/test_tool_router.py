from __future__ import annotations

import pytest

from app.agent.decompose import SubQuery
from app.agent.tool_router import route


def test_route_hybrid_hint():
    sq = SubQuery(text="부가가치세 세율", tool_hint="hybrid")
    assert route(sq) == "hybrid"


def test_route_graph_hint():
    sq = SubQuery(text="판례 사이의 관계", tool_hint="graph")
    assert route(sq) == "graph"


def test_route_keyword_준용():
    sq = SubQuery(text="소득세법 준용 조항", tool_hint="hybrid")
    assert route(sq) == "graph"


def test_route_keyword_관계():
    sq = SubQuery(text="법인세법과 소득세법의 관계", tool_hint="hybrid")
    assert route(sq) == "graph"


def test_route_keyword_개정():
    sq = SubQuery(text="2021년 개정 내용", tool_hint="hybrid")
    assert route(sq) == "graph"


def test_route_plain_query():
    sq = SubQuery(text="부가가치세 신고 기한은?", tool_hint="hybrid")
    assert route(sq) == "hybrid"
