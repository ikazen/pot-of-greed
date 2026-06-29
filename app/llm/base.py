from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: str
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str: ...

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]: ...
