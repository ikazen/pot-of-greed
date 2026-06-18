from __future__ import annotations

import json
import pytest
import respx
import httpx

from app.retrieval.vector_search import Chunk
from app.agent.grounding_check import (
    check_answer, check_claim, apply_grounding, GroundingResult,
)


@pytest.fixture
def ollama_url():
    from app.config import get_settings
    return f"{get_settings().ollama_cloud_base_url}/api/chat"


@pytest.fixture
def sources():
    return [
        Chunk("art_income_14", "article", "소득세법 제14조에 따르면 과세표준은...", 0.9,
              {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True}),
    ]


# ---------------------------------------------------------------------------
# check_answer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_answer_grounded(ollama_url, sources):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": json.dumps({"grounded": True, "issues": []})}},
        ))
        result = await check_answer("소득세법 제14조에 의거하면...", sources)

    assert result.grounded is True
    assert result.issues == []


@pytest.mark.asyncio
async def test_check_answer_not_grounded(ollama_url, sources):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": json.dumps({
                "grounded": False,
                "issues": ["부가가치세 세율 10%라는 주장은 근거에 없음"],
            })}},
        ))
        result = await check_answer("부가가치세 세율은 10%이며...", sources)

    assert result.grounded is False
    assert len(result.issues) == 1
    assert "부가가치세" in result.issues[0]


@pytest.mark.asyncio
async def test_check_answer_fallback_on_error(ollama_url, sources):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(500))
        result = await check_answer("답변", sources)

    assert result.grounded is True


# ---------------------------------------------------------------------------
# check_claim (F2 seam)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_claim_delegates(ollama_url, sources):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200, json={"message": {"content": json.dumps({"grounded": True, "issues": []})}},
        ))
        result = await check_claim("소득세법 제14조에 의거", sources)

    assert result is True


# ---------------------------------------------------------------------------
# apply_grounding
# ---------------------------------------------------------------------------

def test_apply_grounding_pass_when_grounded():
    result = GroundingResult(grounded=True, issues=[])
    assert apply_grounding("정상 답변", result, "flag") == "정상 답변"


def test_apply_grounding_flag_action():
    result = GroundingResult(grounded=False, issues=["환각 주장 A"])
    output = apply_grounding("원본 답변", result, "flag")
    assert "[주의]" in output
    assert "환각 주장 A" in output
    assert "원본 답변" in output


def test_apply_grounding_strip_action():
    result = GroundingResult(grounded=False, issues=["환각 주장이 여기 있습니다"])
    answer = "정상 문장\n환각 주장이 여기 있습니다 — 근거 없음\n또 다른 정상 문장"
    output = apply_grounding(answer, result, "strip")
    assert "환각 주장이 여기 있습니다" not in output
    assert "정상 문장" in output
    assert "또 다른 정상 문장" in output


def test_apply_grounding_empty_issues():
    result = GroundingResult(grounded=False, issues=[])
    assert apply_grounding("원본", result, "flag") == "원본"


# ---------------------------------------------------------------------------
# Integration: grounding affects /chat response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_complex_grounding_flag(async_client, patch_retrieval, patch_llm, monkeypatch):
    """grounded=False, action=flag → 답변 앞에 [주의] 경고."""
    from app.agent.grounding_check import GroundingResult

    async def fake_check_answer_fail(answer, sources):
        return GroundingResult(grounded=False, issues=["근거 없는 주장"])

    monkeypatch.setattr("app.api.chat.check_answer", fake_check_answer_fail)
    monkeypatch.setattr("app.api.chat.apply_grounding", apply_grounding)

    resp = await async_client.post("/chat", json={"query": "법인세 계산?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert "[주의]" in data["answer"]
