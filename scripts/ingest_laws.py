"""
세법 법령 적재 스크립트 — 법제처 OPEN API → PG article_chunks + Neo4j.

사용:
    python scripts/ingest_laws.py

OC 미발급 시 API 호출 부분에서 오류 발생 — fixture 기반 단위 검증만 가능.
OC 발급 후 .env에 LAW_API_OC= 설정 후 실행.
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from neo4j import AsyncGraphDatabase
from pgvector.asyncpg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings
from app.ingest.law_api import fetch_law, list_laws
from app.ingest.law_mapper import MappedLaw, map_law

TARGET_LAWS = ["소득세법", "법인세법", "부가가치세법"]


async def upsert_pg(conn: asyncpg.Connection, mapped: MappedLaw) -> int:
    inserted = 0
    async with conn.transaction():
        for row in mapped.pg_rows:
            result = await conn.execute(
                """
                INSERT INTO article_chunks
                    (chunk_id, law_name, article_no, clause_path, parent_chunk_id,
                     text, effective_from, effective_to, is_current)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                row.chunk_id, row.law_name, row.article_no,
                row.clause_path, row.parent_chunk_id, row.text,
                row.effective_from, row.effective_to, row.is_current,
            )
            if result == "INSERT 0 1":
                inserted += 1
    return inserted


async def upsert_neo4j(session, mapped: MappedLaw) -> None:
    for chunk_id in mapped.neo4j_chunk_ids:
        row = next(r for r in mapped.pg_rows if r.chunk_id == chunk_id)
        await session.run(
            """
            MERGE (n:PotOfGreedArticle {chunk_id: $chunk_id})
            SET n.law_name = $law_name,
                n.article_no = $article_no,
                n.effective_from = $effective_from,
                n.effective_to = $effective_to,
                n.is_current = $is_current
            """,
            chunk_id=chunk_id, law_name=row.law_name, article_no=row.article_no,
            effective_from=str(row.effective_from),
            effective_to=str(row.effective_to) if row.effective_to else None,
            is_current=row.is_current,
        )

    for amend in mapped.amendments:
        await session.run(
            """
            MERGE (n:PotOfGreedAmendment {amendment_id: $amendment_id})
            SET n.law_name = $law_name,
                n.article_no = $article_no,
                n.amended_at = $amended_at,
                n.summary = $summary
            """,
            amendment_id=amend.amendment_id, law_name=amend.law_name,
            article_no=amend.article_no, amended_at=amend.amended_at,
            summary=amend.summary,
        )

    for chunk_id, amend_id in mapped.amended_by:
        await session.run(
            """
            MATCH (a:PotOfGreedArticle {chunk_id: $cid})
            MATCH (m:PotOfGreedAmendment {amendment_id: $mid})
            MERGE (a)-[:AMENDED_BY]->(m)
            """,
            cid=chunk_id, mid=amend_id,
        )


async def ingest_law(law_name: str, pg_conn: asyncpg.Connection, neo4j_session) -> None:
    print(f"[{law_name}] 검색 중...")
    items = await list_laws(law_name)
    current = next((i for i in items if i.is_current), None)
    if not current:
        if items:
            current = items[0]
        else:
            print(f"[{law_name}] 검색 결과 없음, 건너뜀")
            return

    print(f"[{law_name}] MST={current.mst} 조회 중...")
    raw = await fetch_law(current.mst)
    mapped = map_law(raw)

    pg_inserted = await upsert_pg(pg_conn, mapped)
    await upsert_neo4j(neo4j_session, mapped)

    print(
        f"[{law_name}] 완료: PG {len(mapped.pg_rows)}행(신규 {pg_inserted}), "
        f"Neo4j {len(mapped.neo4j_chunk_ids)}노드, "
        f"Amendment {len(mapped.amendments)}건"
    )


async def main() -> None:
    settings = get_settings()
    if not settings.law_api_oc:
        print("LAW_API_OC가 설정되지 않았습니다. .env에 추가 후 재실행하세요.")
        sys.exit(1)

    pg_conn = await asyncpg.connect(dsn=settings.pg_dsn)
    await register_vector(pg_conn)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        async with driver.session() as neo4j_session:
            for law_name in TARGET_LAWS:
                try:
                    await ingest_law(law_name, pg_conn, neo4j_session)
                except Exception as exc:
                    print(f"[{law_name}] 오류: {exc}")

        pg_art = await pg_conn.fetchval("SELECT count(*) FROM article_chunks")
        print(f"\n완료. article_chunks 전체: {pg_art}행")
    finally:
        await pg_conn.close()
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
