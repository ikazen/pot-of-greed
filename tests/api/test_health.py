from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_healthz_ok_when_both_dbs_up(async_client, monkeypatch):
    async def fake_ping_pg():
        return True

    async def fake_ping_neo4j():
        return True

    monkeypatch.setattr("app.api.health.ping_pg", fake_ping_pg)
    monkeypatch.setattr("app.api.health.ping_neo4j", fake_ping_neo4j)

    resp = await async_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"pg": "ok", "neo4j": "ok"}


@pytest.mark.asyncio
async def test_healthz_503_when_pg_down(async_client, monkeypatch):
    async def fake_ping_pg():
        raise RuntimeError("pg unreachable")

    async def fake_ping_neo4j():
        return True

    monkeypatch.setattr("app.api.health.ping_pg", fake_ping_pg)
    monkeypatch.setattr("app.api.health.ping_neo4j", fake_ping_neo4j)

    resp = await async_client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json() == {"pg": "error", "neo4j": "ok"}


@pytest.mark.asyncio
async def test_healthz_no_auth_required(async_client, monkeypatch):
    """#10: /health와 달리 /healthz는 인증 없이 접근 가능해야 한다."""
    import httpx

    async def fake_ping_pg():
        return True

    async def fake_ping_neo4j():
        return True

    monkeypatch.setattr("app.api.health.ping_pg", fake_ping_pg)
    monkeypatch.setattr("app.api.health.ping_neo4j", fake_ping_neo4j)

    from app.main import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:  # Authorization 헤더 없음
        resp = await client.get("/healthz")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_still_requires_auth(monkeypatch):
    """회귀 방지: 기존 /health는 여전히 인증을 요구한다."""
    import httpx
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 401
