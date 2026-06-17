from __future__ import annotations

import os

import httpx
import chainlit as cl

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


@cl.on_message
async def on_message(message: cl.Message):
    user: cl.User = cl.user_session.get("user")
    token = user.metadata["token"]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{API_URL}/chat",
            json={"query": message.content, "mode": "simple"},
            headers={"Authorization": f"Bearer {token}"},
        )

    if resp.status_code != 200:
        await cl.Message(content=f"오류 {resp.status_code}: {resp.text}").send()
        return

    data = resp.json()
    elements: list[cl.Text] = []

    for src in data["sources"]:
        label = src["ref"] or src["chunk_id"]
        kind = "조문" if src["type"] == "article" else "판례"
        elements.append(
            cl.Text(name=label, content=f"[{kind}] {label}", display="inline")
        )

    for warn in data["warnings"]:
        elements.append(
            cl.Text(
                name=f"경고_{warn['ref']}",
                content=f"[주의] {warn['message']}",
                display="inline",
            )
        )

    await cl.Message(content=data["answer"], elements=elements).send()
