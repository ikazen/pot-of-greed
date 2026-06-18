from __future__ import annotations

import json
import pytest
import respx
import httpx

from app.retrieval.vector_search import Chunk
from app.reasoning.answer_builder import (
    build_answer, legal_reasoning_layer,
    _build_warning_message, Warning,
)


# ---------------------------------------------------------------------------
# _build_warning_message — §4 [주의] 포맷
# ---------------------------------------------------------------------------

def test_warning_overruled_has_jujui_prefix():
    msg = _build_warning_message("overruled", {})
    assert msg.startswith("[주의]")
    assert "변경" in msg


def test_warning_law_amended_has_jujui_prefix():
    msg = _build_warning_message("law_amended", {})
    assert msg.startswith("[주의]")
    assert "개정" in msg


def test_warning_law_amended_includes_article_if_available():
    msg = _build_warning_message("law_amended", {"amended_article": "법인세법 제52조"})
    assert "법인세법 제52조" in msg


def test_warning_uncertain_has_jujui_prefix():
    msg = _build_warning_message("uncertain", {})
    assert msg.startswith("[주의]")


# ---------------------------------------------------------------------------
# build_answer — 유효성 경고 병기
# ---------------------------------------------------------------------------

def test_build_answer_overruled_warning():
    chunk = Chunk(
        "case_2019du100", "case", "2019두100 판결...", 0.8,
        {"case_no": "2019두100", "court": "대법원", "validity_flag": "overruled", "decided_at": "2020-01-01"},
    )
    answer = build_answer("판결요지...", [chunk])
    assert len(answer.warnings) == 1
    assert answer.warnings[0].validity_flag == "overruled"
    assert "[주의]" in answer.warnings[0].message


def test_build_answer_law_amended_warning():
    chunk = Chunk(
        "case_2018du12345", "case", "2018두12345 판결...", 0.8,
        {"case_no": "2018두12345", "court": "대법원", "validity_flag": "law_amended", "decided_at": "2019-01-01"},
    )
    answer = build_answer("답변...", [chunk])
    assert answer.warnings[0].validity_flag == "law_amended"
    assert "[주의]" in answer.warnings[0].message


def test_build_answer_valid_case_no_warning():
    chunk = Chunk(
        "case_valid", "case", "유효 판례", 0.8,
        {"case_no": "2020두999", "court": "대법원", "validity_flag": "valid", "decided_at": "2021-01-01"},
    )
    answer = build_answer("답변...", [chunk])
    assert answer.warnings == []


def test_build_answer_article_no_warning():
    chunk = Chunk(
        "art_income_14", "article", "소득세법 제14조...", 0.85,
        {"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True},
    )
    answer = build_answer("답변...", [chunk])
    assert len(answer.sources) == 1
    assert answer.sources[0].type == "article"
    assert answer.warnings == []


def test_build_answer_source_ref_format():
    chunk = Chunk(
        "art_income_14", "article", "소득세법 제14조 제1항...", 0.9,
        {"law_name": "소득세법", "article_no": "제14조", "clause_path": "제1항", "is_current": True},
    )
    answer = build_answer("답변...", [chunk])
    assert "소득세법" in answer.sources[0].ref
    assert "제14조" in answer.sources[0].ref


# ---------------------------------------------------------------------------
# legal_reasoning_layer — 3층 법리 판단
# ---------------------------------------------------------------------------

@pytest.fixture
def ollama_url():
    from app.config import get_settings
    return f"{get_settings().ollama_cloud_base_url}/api/chat"


@pytest.fixture
def chunks_with_warning():
    return [
        Chunk("case_2018du12345", "case", "판결 내용...", 0.8,
              {"case_no": "2018두12345", "court": "대법원", "validity_flag": "law_amended", "decided_at": "2019-01-01"}),
    ]


@pytest.fixture
def warnings():
    return [
        Warning(
            chunk_id="case_2018du12345",
            ref="2018두12345",
            validity_flag="law_amended",
            message="[주의] 근거 조문이 개정됨.",
        )
    ]


@pytest.mark.asyncio
async def test_legal_reasoning_returns_judgment(ollama_url, chunks_with_warning, warnings):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(
            200,
            json={"message": {"content": "법리는 개정 후에도 유지됩니다. 단, 세율 계산 방식은 달라집니다."}},
        ))
        result = await legal_reasoning_layer("법인세 부당행위계산?", chunks_with_warning, warnings)

    assert result is not None
    assert "법리" in result


@pytest.mark.asyncio
async def test_legal_reasoning_returns_none_when_no_warnings(ollama_url, chunks_with_warning):
    result = await legal_reasoning_layer("법인세?", chunks_with_warning, [])
    assert result is None


@pytest.mark.asyncio
async def test_legal_reasoning_fallback_on_error(ollama_url, chunks_with_warning, warnings):
    with respx.mock:
        respx.post(ollama_url).mock(return_value=httpx.Response(500))
        result = await legal_reasoning_layer("법인세?", chunks_with_warning, warnings)

    assert result is None


@pytest.mark.asyncio
async def test_legal_reasoning_uses_1_2_layer_context(ollama_url, chunks_with_warning, warnings):
    """1층(validity_flag) 정보가 LLM 요청 body에 포함되는지 확인."""
    captured_body = {}

    def capture(request):
        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "법리 판단 결과"}})

    with respx.mock:
        respx.post(ollama_url).mock(side_effect=capture)
        await legal_reasoning_layer("법인세?", chunks_with_warning, warnings)

    user_msg = captured_body["messages"][1]["content"]
    assert "[주의]" in user_msg or "law_amended" in user_msg or "개정" in user_msg


# ---------------------------------------------------------------------------
# Integration: 3층 판단이 /chat complex 응답에 포함
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_complex_includes_legal_reasoning(async_client, patch_retrieval, patch_llm, monkeypatch):
    """복잡 모드 + law_amended 판례 → 답변에 '법리 검토' 섹션 포함."""
    from app.retrieval.vector_search import Chunk
    from app.agent.grounding_check import GroundingResult

    amended_chunk = Chunk(
        "case_amended", "case", "판례 내용", 0.8,
        {"case_no": "2018두12345", "court": "대법원", "validity_flag": "law_amended", "decided_at": "2019-01-01"},
    )

    async def fake_check_answer(answer, sources):
        return GroundingResult(grounded=True)

    def fake_apply_grounding(raw, result, action):
        return raw

    async def fake_legal_reasoning(query, chunks, warnings):
        if warnings:
            return "개정 이후에도 법리는 유지됩니다."
        return None

    # patch_retrieval already patches everything, just override legal_reasoning_layer
    monkeypatch.setattr("app.api.chat.check_answer", fake_check_answer)
    monkeypatch.setattr("app.api.chat.apply_grounding", fake_apply_grounding)
    monkeypatch.setattr("app.api.chat.legal_reasoning_layer", fake_legal_reasoning)

    # override rerank to return the amended chunk
    async def fake_rerank_amended(query, chunks, top_k=None):
        return [amended_chunk]

    monkeypatch.setattr("app.api.chat.rerank", fake_rerank_amended)

    resp = await async_client.post("/chat", json={"query": "법인세 판례?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert "법리 검토" in data["answer"]
    assert "개정 이후에도" in data["answer"]


# ---------------------------------------------------------------------------
# _extract_transaction_date — 2층 시점 필터
# ---------------------------------------------------------------------------

def test_extract_transaction_date_iso():
    from app.api.chat import _extract_transaction_date
    assert _extract_transaction_date("2018-06-01 거래") == "2018-06-01"


def test_extract_transaction_date_korean():
    from app.api.chat import _extract_transaction_date
    result = _extract_transaction_date("2018년 6월 15일 기준")
    assert result == "2018-06-15"


def test_extract_transaction_date_month_only():
    from app.api.chat import _extract_transaction_date
    result = _extract_transaction_date("2021년 3월 개정")
    assert result == "2021-03-01"


def test_extract_transaction_date_none():
    from app.api.chat import _extract_transaction_date
    assert _extract_transaction_date("일반적인 질문") is None
