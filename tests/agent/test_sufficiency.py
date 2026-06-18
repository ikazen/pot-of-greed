from __future__ import annotations

import json
import time
import pytest
import respx
import httpx

from app.retrieval.vector_search import Chunk
from app.agent.sufficiency import evaluate, sufficiency_loop, SufficiencyResult


@pytest.fixture
def ollama_url():
    from app.config import get_settings
    return f"{get_settings().ollama_cloud_base_url}/api/chat"


@pytest.fixture
def sample_chunks():
    return [
        Chunk("art_income_14", "article", "소득세법 제14조...", 0.8,
              {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True}),
    ]


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_sufficient(ollama_url, sample_chunks):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": json.dumps({"sufficient": True})}},
        ))
        result = await evaluate("소득세 과세표준?", sample_chunks)

    assert result.sufficient is True
    assert result.rewritten_query is None


@pytest.mark.asyncio
async def test_evaluate_insufficient(ollama_url, sample_chunks):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": json.dumps(
                {"sufficient": False, "rewritten_query": "소득세법 제14조 과세표준 계산 방법"}
            )}},
        ))
        result = await evaluate("소득세?", sample_chunks)

    assert result.sufficient is False
    assert result.rewritten_query == "소득세법 제14조 과세표준 계산 방법"


@pytest.mark.asyncio
async def test_evaluate_fallback_on_error(ollama_url, sample_chunks):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(500))
        result = await evaluate("세금?", sample_chunks)

    assert result.sufficient is True


# ---------------------------------------------------------------------------
# sufficiency_loop()
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_settings():
    from app.config import get_settings
    return get_settings()


@pytest.mark.asyncio
async def test_loop_exits_on_sufficient(ollama_url, sample_chunks, fake_settings):
    call_count = 0

    async def retrieve_fn(q):
        nonlocal call_count
        call_count += 1
        return sample_chunks

    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": json.dumps({"sufficient": True})}},
        ))
        deadline = time.monotonic() + 20
        result = await sufficiency_loop("소득세?", retrieve_fn, fake_settings, deadline)

    assert call_count == 1
    assert result == sample_chunks


@pytest.mark.asyncio
async def test_loop_reruns_on_insufficient(ollama_url, sample_chunks, fake_settings):
    """부족 판정 1회 → 재검색 후 충분 판정."""
    call_count = 0
    responses = [
        json.dumps({"sufficient": False, "rewritten_query": "소득세법 제14조 상세"}),
        json.dumps({"sufficient": True}),
    ]

    async def retrieve_fn(q):
        nonlocal call_count
        call_count += 1
        return sample_chunks

    with respx.mock:
        respx.post(ollama_url).mock(side_effect=[
            httpx.Response(200, json={"message": {"content": responses[0]}}),
            httpx.Response(200, json={"message": {"content": responses[1]}}),
        ])
        deadline = time.monotonic() + 20
        result = await sufficiency_loop("소득세?", retrieve_fn, fake_settings, deadline)

    assert call_count == 2
    assert result == sample_chunks


@pytest.mark.asyncio
async def test_loop_max_iter_cap(ollama_url, sample_chunks, fake_settings):
    """max_iter(=2) 도달 시 무한루프 없이 종료."""
    call_count = 0

    async def retrieve_fn(q):
        nonlocal call_count
        call_count += 1
        return sample_chunks

    always_insufficient = json.dumps({"sufficient": False, "rewritten_query": "다시"})

    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": always_insufficient}},
        ))
        deadline = time.monotonic() + 20
        result = await sufficiency_loop("애매한 질의", retrieve_fn, fake_settings, deadline)

    # max_iter=2이면 retrieve_fn은 최초 1회 + 재검색 최대 2회 = 3회 이하
    assert call_count <= fake_settings.sufficiency_max_iter + 1


@pytest.mark.asyncio
async def test_loop_deadline_exit(sample_chunks, fake_settings):
    """deadline 초과 시 조기 탈출 (LLM 호출 없이 결과 반환)."""
    call_count = 0

    async def retrieve_fn(q):
        nonlocal call_count
        call_count += 1
        return sample_chunks

    past_deadline = time.monotonic() - 1.0  # 이미 지난 deadline
    result = await sufficiency_loop("질의", retrieve_fn, fake_settings, past_deadline)

    assert result == sample_chunks
    assert call_count == 1  # 최초 검색은 수행, 루프 진입 즉시 탈출
