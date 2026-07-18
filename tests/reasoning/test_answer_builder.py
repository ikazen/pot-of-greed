from __future__ import annotations

import json
import pytest
import respx
import httpx

from app.retrieval.vector_search import Chunk
from app.reasoning.answer_builder import (
    legal_reasoning_layer,
    build_warning_message, first_line, Warning,
)


# ---------------------------------------------------------------------------
# build_warning_message — §4 [주의] 포맷
# ---------------------------------------------------------------------------

def test_warning_overruled_has_jujui_prefix():
    msg = build_warning_message("overruled", {})
    assert msg.startswith("[주의]")
    assert "변경" in msg


def test_warning_law_amended_has_jujui_prefix():
    msg = build_warning_message("law_amended", {})
    assert msg.startswith("[주의]")
    assert "개정" in msg


def test_warning_law_amended_includes_article_if_available():
    msg = build_warning_message("law_amended", {"amended_article": "법인세법 제52조"})
    assert "법인세법 제52조" in msg


def test_warning_uncertain_has_jujui_prefix():
    msg = build_warning_message("uncertain", {})
    assert msg.startswith("[주의]")


# ---------------------------------------------------------------------------
# first_line — 요약 추출
# ---------------------------------------------------------------------------

def test_first_line_returns_first_nonempty():
    assert first_line("\n\n판시사항 내용\n두번째 줄") == "판시사항 내용"


def test_first_line_truncates_at_limit():
    long = "a" * 200
    result = first_line(long, limit=100)
    assert len(result) == 103  # 100 chars + "..."
    assert result.endswith("...")


def test_first_line_no_truncation_when_short():
    result = first_line("짧은 요약", limit=100)
    assert result == "짧은 요약"


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
async def test_chat_complex_includes_legal_reasoning(async_client, patch_retrieval, patch_rarr, monkeypatch):
    """RARR 경로: complex 모드 응답 계약 검증 (법리 검토는 pipeline 내부 담당)."""
    resp = await async_client.post("/chat", json={"query": "법인세 판례?", "mode": "complex"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["warnings"], list)


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
