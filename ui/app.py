from __future__ import annotations

import json
import os

import httpx
import chainlit as cl

import data_layer  # noqa: F401 — registers @cl.data_layer via side-effect

API_URL = os.environ.get("API_BASE_URL", "http://pot-of-greed-api:8000")


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{API_URL}/auth/token",
            data={"username": username, "password": password},
        )
    if resp.status_code != 200:
        return None
    token = resp.json()["access_token"]
    return cl.User(identifier=username, metadata={"token": token})


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass


@cl.on_message
async def on_message(message: cl.Message):
    user: cl.User = cl.user_session.get("user")
    token = user.metadata["token"]

    msg = cl.Message(content="")
    await msg.send()

    sources_data: list[dict] = []
    warnings_data: list[dict] = []
    debug_data: dict | None = None

    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream(
            "POST",
            f"{API_URL}/chat/stream",
            json={"query": message.content, "mode": "simple"},
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            if resp.status_code != 200:
                await cl.Message(content=f"오류 {resp.status_code}").send()
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if "token" in data:
                    await msg.stream_token(data["token"])
                elif "sources" in data:
                    sources_data = data["sources"]
                    warnings_data = data["warnings"]
                    debug_data = data.get("debug")

    extra: list[str] = []

    if warnings_data:
        warn_lines = "\n".join(f"> {w['message']}" for w in warnings_data)
        extra.append(warn_lines)

    if sources_data:
        lines = ["\n---\n참고 출처"]
        for src in sources_data:
            label = src["ref"] or src["chunk_id"]
            kind = "조문" if src["type"] == "article" else "판례"
            summary = src.get("summary", "")
            lines.append(f"- [{kind}] {label}")
            if summary:
                lines.append(f"  {summary}")
        extra.append("\n".join(lines))

    if debug_data:
        extra.append(_render_debug(debug_data))

    if extra:
        msg.content += "\n\n" + "\n\n".join(extra)
        await msg.update()


def _render_debug(debug: dict) -> str:
    """RARR 파이프라인 수정 내역 — 디버그 모드(DEBUG_PIPELINE=true)일 때만 전달됨."""
    lines = ["<details>", "<summary>파이프라인 수정 내역 (디버그)</summary>", ""]
    lines.append(
        f"mode={debug.get('mode')} claims={debug.get('claims_total', 0)} "
        f"deferred={debug.get('deferred_count', 0)} "
        f"법리검토={'적용' if debug.get('legal_reasoning_applied') else '미적용'}"
    )
    scrubbed = debug.get("scrubbed_refs") or []
    if scrubbed:
        lines.append(f"삭제된 환각 인용: {', '.join(scrubbed)}")

    for i, claim in enumerate(debug.get("claims", []), start=1):
        lines.append(f"\n**주장 {i}**")
        lines.append(f"- 원문: {claim['original']}")
        verdict = "일치" if claim["agree"] else "불일치"
        reason = claim.get("agreement_reason") or ""
        lines.append(f"- 판정: {verdict}" + (f" ({reason})" if reason else ""))
        if claim.get("revised"):
            lines.append(f"- 수정: {claim['revised']}")
        if claim.get("corrections"):
            lines.append(f"- 정정: {', '.join(claim['corrections'])}")
        if claim.get("removed_refs"):
            lines.append(f"- 삭제된 인용: {', '.join(claim['removed_refs'])}")
        refs = claim.get("evidence_refs") or []
        lines.append(f"- 근거({claim.get('evidence_count', 0)}): {', '.join(refs) if refs else '없음'}")

    lines.append("</details>")
    return "\n".join(lines)
