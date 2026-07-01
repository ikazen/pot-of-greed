from __future__ import annotations

import pytest


class _FakeConn:
    """article_chunks/case_chunks 동등 매칭을 시뮬레이션하는 fake connection.

    시드는 (law_name, article_no) 튜플 집합과 case_no 집합으로 구성.
    실행된 (sql, params)를 기록해 어떤 컬럼으로 질의했는지 assert 가능.
    """

    def __init__(self, articles: set[tuple[str, str]], cases: set[str]):
        self.articles = articles
        self.cases = cases
        self.calls: list[tuple[str, tuple]] = []

    async def fetchval(self, sql: str, *params):
        self.calls.append((sql, params))
        if "article_chunks" in sql:
            return 1 if tuple(params) in self.articles else None
        return 1 if params[0] in self.cases else None


class _FakeAcquire:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


def _patch_pool(monkeypatch, articles=frozenset(), cases=frozenset()):
    conn = _FakeConn(set(articles), set(cases))
    pool = _FakePool(conn)

    import app.rarr.citation as citation_mod
    monkeypatch.setattr(citation_mod, "get_pool", lambda: pool)
    return conn


@pytest.mark.asyncio
async def test_verify_citations_existing(monkeypatch):
    _patch_pool(monkeypatch, articles={("소득세법", "제89조")})

    from app.rarr.citation import verify_citations
    result = await verify_citations(["소득세법 제89조"])
    assert result["소득세법 제89조"] is True


@pytest.mark.asyncio
async def test_verify_citations_hallucinated(monkeypatch):
    _patch_pool(monkeypatch)

    from app.rarr.citation import verify_citations
    result = await verify_citations(["존재하지않는법 제9999조"])
    assert result["존재하지않는법 제9999조"] is False


@pytest.mark.asyncio
async def test_verify_citations_wrong_law_name_rejected(monkeypatch):
    """법명은 오귀속이고 조번호만 실재하는 인용 — 구조적 동등 매칭에서 걸러져야 한다.

    회귀 방지: 기존 FTS AND-토큰 매칭이었다면 두 토큰이 다른 청크에서라도
    동시 등장하면 통과했던 케이스.
    """
    _patch_pool(monkeypatch, articles={("소득세법", "제89조")})

    from app.rarr.citation import verify_citations
    result = await verify_citations(["법인세법 제89조"])
    assert result["법인세법 제89조"] is False


@pytest.mark.asyncio
async def test_verify_citations_empty_refs():
    from app.rarr.citation import verify_citations
    result = await verify_citations([])
    assert result == {}


@pytest.mark.asyncio
async def test_verify_citations_multiple(monkeypatch):
    _patch_pool(monkeypatch, articles={("소득세법", "제89조")})

    from app.rarr.citation import verify_citations
    result = await verify_citations(["소득세법 제89조", "없는법 제1조"])
    assert result["소득세법 제89조"] is True
    assert result["없는법 제1조"] is False


@pytest.mark.asyncio
async def test_verify_citations_case_no(monkeypatch):
    _patch_pool(monkeypatch, cases={"2018두12345"})

    from app.rarr.citation import verify_citations
    result = await verify_citations(["2018두12345"])
    assert result["2018두12345"] is True


@pytest.mark.asyncio
async def test_verify_citations_clause_matched_at_article_level(monkeypatch):
    """항(제N항)이 붙은 ref도 조 단위(law_name+article_no)로 매칭된다."""
    conn = _patch_pool(monkeypatch, articles={("소득세법", "제14조")})

    from app.rarr.citation import verify_citations
    result = await verify_citations(["소득세법 제14조 제1항"])
    assert result["소득세법 제14조 제1항"] is True
    assert conn.calls == [
        (
            "SELECT 1 FROM article_chunks WHERE law_name = $1 AND article_no = $2 LIMIT 1",
            ("소득세법", "제14조"),
        )
    ]


class TestParseRef:
    def test_article(self):
        from app.rarr.claims import parse_ref
        assert parse_ref("소득세법 제89조") == ("article", ("소득세법", "제89조"))

    def test_article_with_clause(self):
        from app.rarr.claims import parse_ref
        assert parse_ref("소득세법 제14조 제1항") == ("article", ("소득세법", "제14조"))

    def test_case(self):
        from app.rarr.claims import parse_ref
        assert parse_ref("2018두12345") == ("case", ("2018두12345",))

    def test_unparseable(self):
        from app.rarr.claims import parse_ref
        assert parse_ref("아무말") is None
