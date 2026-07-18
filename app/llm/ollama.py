from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable

import httpx

from app.llm.base import Message


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        default_timeout: float = 120.0,
        on_request: Callable[[dict], None] | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._default_timeout = default_timeout
        self._on_request = on_request
        # #19: 호출마다 새 AsyncClient를 열면 매번 콜드 TLS 핸드셰이크가 발생해
        # 동시 요청 부하에서 ReadTimeout이 잦았다. 인스턴스 수명 동안 재사용.
        self._client = httpx.AsyncClient(timeout=default_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

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

    def _fire_hook(self, payload: dict) -> None:
        if self._on_request:
            headers = self._headers()
            # API 키는 마스킹
            redacted = {k: "***" if k.lower() in ("authorization",) else v for k, v in headers.items()}
            self._on_request({
                "transport": f"POST {self._base_url}/api/chat",
                "headers": redacted,
                "body": payload,
            })

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

        self._fire_hook(payload)

        resp = await self._client.post(
            f"{self._base_url}/api/chat",
            headers=self._headers(),
            json=payload,
            timeout=timeout if timeout is not None else self._default_timeout,
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

        self._fire_hook(payload)

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            headers=self._headers(),
            json=payload,
            timeout=timeout if timeout is not None else self._default_timeout,
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
