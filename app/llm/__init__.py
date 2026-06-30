from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from app.llm.base import LLMProvider, Message
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider


def _build_provider(
    settings,
    *,
    on_request: Callable[[dict], None] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    name = provider or settings.llm_provider
    if name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=model or settings.gemini_model,
            default_timeout=float(settings.llm_timeout_s),
            on_request=on_request,
        )
    return OllamaProvider(
        base_url=settings.ollama_cloud_base_url,
        model=model or settings.llm_model,
        api_key=settings.ollama_api_key,
        default_timeout=float(settings.llm_timeout_s),
        on_request=on_request,
    )


def _role_provider_model(settings, role: str) -> tuple[str, str] | None:
    """role → (provider, model). 알 수 없는 role이면 None."""
    return {
        "draft": ("gemini", settings.rarr_draft_model),
        "edit": ("gemini", settings.rarr_edit_model),
        "reason": ("gemini", settings.rarr_reason_model),
        "aux": (settings.rarr_aux_provider, settings.rarr_aux_model),
    }.get(role)


@lru_cache
def get_llm_provider(role: str = "default") -> LLMProvider:
    from app.config import get_settings
    settings = get_settings()
    mapping = _role_provider_model(settings, role)
    if mapping is not None:
        provider, model = mapping
        return _build_provider(settings, provider=provider, model=model)
    return _build_provider(settings)


def make_llm_provider(
    *,
    on_request: Callable[[dict], None] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """캐시 우회 인스턴스 생성. CLI/테스트에서 훅·오버라이드 주입용."""
    from app.config import get_settings
    return _build_provider(get_settings(), on_request=on_request, provider=provider, model=model)


__all__ = ["get_llm_provider", "make_llm_provider", "LLMProvider", "Message"]
