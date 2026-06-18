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

    elements: list[cl.Text] = []
    for src in sources_data:
        label = src["ref"] or src["chunk_id"]
        kind = "조문" if src["type"] == "article" else "판례"
        elements.append(
            cl.Text(name=label, content=f"[{kind}] {label}", display="inline")
        )
    for warn in warnings_data:
        elements.append(
            cl.Text(
                name=f"경고_{warn['ref']}",
                content=f"[주의] {warn['message']}",
                display="inline",
            )
        )

    if elements:
        msg.elements = elements
        await msg.update()
