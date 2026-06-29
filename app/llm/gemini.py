from __future__ import annotations

from collections.abc import AsyncGenerator

import google.genai as genai
import google.genai.types as t

from app.llm.base import Message

# Gemini role 매핑: messages의 "assistant" → Gemini "model"
_ROLE_MAP = {"user": "user", "assistant": "model"}


def _to_contents(messages: list[Message]) -> list[t.Content]:
    return [
        t.Content(role=_ROLE_MAP.get(m["role"], m["role"]), parts=[t.Part(text=m["content"])])
        for m in messages
    ]


class GeminiProvider:
    def __init__(self, api_key: str, model: str, default_timeout: float = 120.0) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._default_timeout = default_timeout

    def _config(
        self,
        system: str | None,
        json_mode: bool,
        timeout: float | None,
    ) -> t.GenerateContentConfig:
        kwargs: dict = {}
        if system:
            kwargs["system_instruction"] = system
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        http_opts: dict = {}
        if timeout is not None:
            http_opts["timeout"] = timeout
        elif self._default_timeout:
            http_opts["timeout"] = self._default_timeout
        if http_opts:
            kwargs["http_options"] = t.HttpOptions(**http_opts)
        return t.GenerateContentConfig(**kwargs)

    async def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str:
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=_to_contents(messages),
            config=self._config(system, json_mode, timeout),
        )
        return resp.text or ""

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=_to_contents(messages),
            config=self._config(system, False, timeout),
        ):
            text = chunk.text
            if text:
                yield text
