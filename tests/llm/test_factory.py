from __future__ import annotations

import pytest

from app.llm import get_llm_provider
from app.llm.ollama import OllamaProvider
from app.llm.gemini import GeminiProvider


@pytest.mark.asyncio
async def test_factory_returns_ollama_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    from app.config import get_settings
    get_settings.cache_clear()
    get_llm_provider.cache_clear()

    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)

    get_llm_provider.cache_clear()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_factory_returns_gemini_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings
    get_settings.cache_clear()
    get_llm_provider.cache_clear()

    provider = get_llm_provider()
    assert isinstance(provider, GeminiProvider)

    get_llm_provider.cache_clear()
    get_settings.cache_clear()
