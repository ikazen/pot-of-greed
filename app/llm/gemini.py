from __future__ import annotations

from collections.abc import AsyncGenerator, Callable

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
    def __init__(
        self,
        api_key: str,
        model: str,
        default_timeout: float = 120.0,
        on_request: Callable[[dict], None] | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._default_timeout = default_timeout
        self._on_request = on_request

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
        # HttpOptions.timeout 단위는 ms
        http_opts: dict = {}
        if timeout is not None:
            http_opts["timeout"] = int(timeout * 1000)
        elif self._default_timeout:
            http_opts["timeout"] = int(self._default_timeout * 1000)
        if http_opts:
            kwargs["http_options"] = t.HttpOptions(**http_opts)
        return t.GenerateContentConfig(**kwargs)

    def _fire_hook(self, contents: list[t.Content], config: t.GenerateContentConfig) -> None:
        if not self._on_request:
            return
        self._on_request({
            "transport": "google-genai SDK → generateContent",
            "model": self._model,
            "contents": [c.model_dump(exclude_none=True) for c in contents],
            "config": config.model_dump(exclude_none=True),
        })

    async def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str:
        contents = _to_contents(messages)
        config = self._config(system, json_mode, timeout)
        self._fire_hook(contents, config)
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return resp.text or ""

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        contents = _to_contents(messages)
        config = self._config(system, False, timeout)
        self._fire_hook(contents, config)
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            text = chunk.text
            if text:
                yield text
