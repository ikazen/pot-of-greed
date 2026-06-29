from __future__ import annotations

from functools import lru_cache

from app.llm.base import LLMProvider, Message
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    from app.config import get_settings
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            default_timeout=float(settings.llm_timeout_s),
        )
    return OllamaProvider(
        base_url=settings.ollama_cloud_base_url,
        model=settings.llm_model,
        api_key=settings.ollama_api_key,
        default_timeout=float(settings.llm_timeout_s),
    )


__all__ = ["get_llm_provider", "LLMProvider", "Message"]
