from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx

from app.llm.base import Message


class OllamaProvider:
    def __init__(self, base_url: str, model: str, api_key: str = "", default_timeout: float = 120.0) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._default_timeout = default_timeout

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def _build_messages(self, messages: list[Message], system: str | None) -> list[dict]:
        built: list[dict] = []
        if system:
            built.append({"role": "system", "content": system})
        built.extend(messages)
        return built

    async def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "messages": self._build_messages(messages, system),
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=timeout or self._default_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self._model,
            "messages": self._build_messages(messages, system),
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=timeout or self._default_timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk_data = json.loads(line)
                    token = chunk_data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk_data.get("done"):
                        break
