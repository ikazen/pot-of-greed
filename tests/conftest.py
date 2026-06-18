from __future__ import annotations

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
}


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    for k, v in _TEST_ENV.items():
        monkeypatch.setenv(k, v)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
        return chunks[:top_k] if top_k else chunks

    async def fake_expand_1hop(chunk_ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
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
        return chunks[:top_k] if top_k else chunks

    async def fake_expand_1hop(chunk_ids):
        return []

    async def fake_expand_to_parents(chunks):
        return []

    monkeypatch.setattr("app.api.chat.embed_query", fake_embed_query)
    monkeypatch.setattr("app.api.chat.vector_search", fake_vector_search)
    monkeypatch.setattr("app.api.chat.keyword_search", fake_keyword_search)
    monkeypatch.setattr("app.api.chat.rerank", fake_rerank)
    monkeypatch.setattr("app.api.chat.expand_1hop", fake_expand_1hop)
    monkeypatch.setattr("app.api.chat.expand_to_parents", fake_expand_to_parents)
    return low_score_chunks


@pytest.fixture
def patch_llm(monkeypatch):
    async def fake_simple_inference(query, chunks):
        return "단순 모드 답변입니다."

    async def fake_complex_inference(query, chunks, system_extra=""):
        return "복잡 모드 답변입니다."

    async def fake_stream(query, chunks):
        for token in ["스트리밍 ", "답변입니다."]:
            yield token

    monkeypatch.setattr("app.api.chat.simple_inference", fake_simple_inference)
    monkeypatch.setattr("app.api.chat.complex_inference", fake_complex_inference)
    monkeypatch.setattr("app.api.chat.stream_simple_inference", fake_stream)


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
