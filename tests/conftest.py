from __future__ import annotations

from dataclasses import replace

import pytest
import httpx

from app.retrieval.vector_search import Chunk

# ---------------------------------------------------------------------------
# Required env vars for Settings
# ---------------------------------------------------------------------------

_TEST_ENV = {
    "PG_DSN": "postgresql://test:test@localhost/test",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_PASSWORD": "test",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_CLOUD_BASE_URL": "http://localhost:11435",
    "JWT_SECRET": "test-secret-key",
    "AUTH_USERS": "testuser:$2b$12$FAKEHASHFORTESTINGwwwwuO",
    "LAW_API_OC": "test-oc",
    # 테스트는 ollama provider 고정 — respx로 /api/chat URL 직접 모킹하는 기존 테스트 호환
    "LLM_PROVIDER": "ollama",
    "GEMINI_API_KEY": "",
    # RARR 역할도 ollama로 통일 (respx mock 호환)
    "RARR_DRAFT_PROVIDER": "ollama",
    "RARR_EDIT_PROVIDER": "ollama",
    "RARR_REASON_PROVIDER": "ollama",
}


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    for k, v in _TEST_ENV.items():
        monkeypatch.setenv(k, v)
    from app.config import get_settings
    from app.llm import get_llm_provider
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_provider.cache_clear()


# ---------------------------------------------------------------------------
# Sample Chunks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_article_chunk():
    return Chunk(
        chunk_id="art_income_14",
        table="article",
        text="소득세법 제14조 과세표준의 계산...",
        score=0.85,
        meta={"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True},
    )


@pytest.fixture
def sample_case_chunk():
    return Chunk(
        chunk_id="case_2018du12345",
        table="case",
        text="대법원 2018두12345 판결...",
        score=0.75,
        meta={"case_no": "2018두12345", "court": "대법원", "validity_flag": "valid", "decided_at": "2019-01-01"},
    )


@pytest.fixture
def sample_chunks(sample_article_chunk, sample_case_chunk):
    return [sample_article_chunk, sample_case_chunk]


@pytest.fixture
def low_score_chunks():
    return [
        Chunk(
            chunk_id="art_income_14",
            table="article",
            text="소득세법 제14조...",
            score=0.3,
            meta={"law_name": "소득세법", "article_no": "제14조", "clause_path": None, "is_current": True},
        )
    ]


# ---------------------------------------------------------------------------
# Retrieval patches
# Patch in app.api.chat where names are imported (not source modules).
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_retrieval(monkeypatch, sample_chunks):
    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return sample_chunks

    async def fake_keyword_search(query, top_k=30):
        return sample_chunks

    async def fake_rerank(query, chunks, top_k=None):
        # 실제 reranker는 항상 relevance score로 .score를 재할당한다(reranker.py).
        # RRF 융합 후 score가 RRF 스케일로 바뀌므로(#9) 여기서도 재할당해야
        # should_promote 임계값(0.5) 비교가 원래 의도한 스케일로 맞는다.
        result = chunks[:top_k] if top_k else chunks
        return [replace(c, score=0.85) for c in result]

    async def fake_expand_1hop(chunk_ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    async def fake_hyde_embedding(query):
        return [0.2] * 1024

    async def fake_decompose(query):
        from app.agent.decompose import SubQuery
        return [SubQuery(text=query, tool_hint="hybrid")]

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    monkeypatch.setattr("app.api.chat.hyde_embedding", fake_hyde_embedding)
    monkeypatch.setattr("app.api.chat.decompose", fake_decompose)
    return sample_chunks


@pytest.fixture
def patch_low_score_retrieval(monkeypatch, low_score_chunks):
    async def fake_embed_query(text):
        return [0.1] * 1024

    async def fake_vector_search(embedding, top_k=30, only_current=True):
        return low_score_chunks

    async def fake_keyword_search(query, top_k=30):
        return low_score_chunks

    async def fake_rerank(query, chunks, top_k=None):
        result = chunks[:top_k] if top_k else chunks
        return [replace(c, score=0.3) for c in result]

    async def fake_expand_1hop(chunk_ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    async def fake_hyde_embedding(query):
        return [0.2] * 1024

    async def fake_decompose(query):
        from app.agent.decompose import SubQuery
        return [SubQuery(text=query, tool_hint="hybrid")]

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    monkeypatch.setattr("app.api.chat.hyde_embedding", fake_hyde_embedding)
    monkeypatch.setattr("app.api.chat.decompose", fake_decompose)
    return low_score_chunks


@pytest.fixture
def patch_rarr(monkeypatch):
    from app.rarr.pipeline import RarrResult
    from app.reasoning.answer_builder import Source

    _SIMPLE_ANSWER = "단순 모드 RARR 답변입니다."
    _COMPLEX_ANSWER = "복잡 모드 RARR 답변입니다."
    _SAMPLE_SOURCE = Source(
        type="article",
        ref="소득세법 제14조",
        chunk_id="art_income_14",
        summary="과세표준의 계산...",
    )

    async def fake_run_rarr(query, mode, settings, on_progress=None):
        if on_progress:
            on_progress("검증 완료")
        answer = _COMPLEX_ANSWER if mode == "complex" else _SIMPLE_ANSWER
        return RarrResult(answer=answer, sources=[_SAMPLE_SOURCE], warnings=[], attributions=[])

    monkeypatch.setattr("app.api.chat.run_rarr", fake_run_rarr)


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_token():
    from app.config import get_settings
    from app.auth.jwt import create_token
    return create_token("testuser", get_settings())


@pytest.fixture
async def async_client(auth_token):
    from app.main import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {auth_token}"},
    ) as client:
        yield client
