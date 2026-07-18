from __future__ import annotations

import pytest

from app.llm import get_llm_provider
from app.llm.ollama import OllamaProvider
from app.llm.gemini import GeminiProvider


def _clear():
    from app.config import get_settings
    get_settings.cache_clear()
    get_llm_provider.cache_clear()


@pytest.fixture(autouse=True)
def clear_caches():
    _clear()
    yield
    _clear()


def test_default_role_uses_global_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _clear()
    assert isinstance(get_llm_provider(), GeminiProvider)


def test_no_arg_equals_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _clear()
    # 무인자 호출과 role="default" 는 각각 캐시되지만 동일 타입 반환
    assert type(get_llm_provider()) is type(get_llm_provider("default"))


def test_draft_role_returns_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RARR_DRAFT_PROVIDER", "gemini")
    monkeypatch.setenv("RARR_DRAFT_MODEL", "gemini-2.5-flash")
    _clear()
    provider = get_llm_provider("draft")
    assert isinstance(provider, GeminiProvider)


def test_edit_role_returns_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RARR_EDIT_PROVIDER", "gemini")
    _clear()
    assert isinstance(get_llm_provider("edit"), GeminiProvider)


def test_reason_role_returns_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RARR_REASON_PROVIDER", "gemini")
    _clear()
    assert isinstance(get_llm_provider("reason"), GeminiProvider)


def test_aux_role_returns_ollama(monkeypatch):
    monkeypatch.setenv("RARR_AUX_PROVIDER", "ollama")
    monkeypatch.setenv("RARR_AUX_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.example.com")
    _clear()
    provider = get_llm_provider("aux")
    assert isinstance(provider, OllamaProvider)


def test_aux_model_tag(monkeypatch):
    monkeypatch.setenv("RARR_AUX_PROVIDER", "ollama")
    monkeypatch.setenv("RARR_AUX_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.example.com")
    _clear()
    provider = get_llm_provider("aux")
    assert isinstance(provider, OllamaProvider)
    assert provider._model == "gpt-oss:20b"


def test_per_role_cache(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RARR_AUX_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.example.com")
    _clear()
    draft1 = get_llm_provider("draft")
    draft2 = get_llm_provider("draft")
    aux1 = get_llm_provider("aux")
    assert draft1 is draft2
    assert draft1 is not aux1


def test_unknown_role_falls_back_to_global(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _clear()
    provider = get_llm_provider("nonexistent-role")
    assert isinstance(provider, GeminiProvider)
