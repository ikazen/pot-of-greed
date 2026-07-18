from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_login_success_returns_token(monkeypatch):
    """#10: authenticate_user가 asyncio.to_thread로 옮겨져도 성공 경로는 동일해야 한다."""
    def fake_authenticate_user(username, password, settings):
        return username == "testuser" and password == "correct"

    monkeypatch.setattr("app.auth.routes.authenticate_user", fake_authenticate_user)

    from app.main import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/auth/token", data={"username": "testuser", "password": "correct"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401(monkeypatch):
    def fake_authenticate_user(username, password, settings):
        return False

    monkeypatch.setattr("app.auth.routes.authenticate_user", fake_authenticate_user)

    from app.main import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/auth/token", data={"username": "testuser", "password": "wrong"}
        )
    assert resp.status_code == 401
