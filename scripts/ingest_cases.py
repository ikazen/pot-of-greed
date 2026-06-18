"""
세법 판례 적재 스크립트 — 법제처 OPEN API → PG case_chunks + Neo4j.

스코핑: 참조조문이 적재된 세법 article을 가리키는 판례만 수입.
적재 후 scripts/update_validity.py 실행으로 validity_flag 갱신 필요.

사용:
    python scripts/ingest_cases.py
    python scripts/update_validity.py  # 이후 반드시 실행
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from neo4j import AsyncGraphDatabase
from pgvector.asyncpg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings
from app.ingest.case_mapper import MappedCase, map_case
from app.ingest.law_api import fetch_case, list_cases

# 세법 관련 검색어 — 참조조문 스코프 필터로 1차 보완됨
SEARCH_QUERIES = ["소득세", "법인세", "부가가치세"]
MAX_PAGES_PER_QUERY = 50  # 쿼리당 최대 페이지 (= 최대 1000건)


async def load_known_article_ids(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT chunk_id FROM article_chunks")
    return {r["chunk_id"] for r in rows}


async def upsert_pg(conn: asyncpg.Connection, mapped: MappedCase) -> bool:
    r = mapped.case_row
    result = await conn.execute(
        """
        INSERT INTO case_chunks
            (chunk_id, case_no, court, decided_at, is_en_banc, validity_flag, text)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (chunk_id) DO NOTHING
        """,
        r.chunk_id, r.case_no, r.court, r.decided_at,
        r.is_en_banc, r.validity_flag, r.text,
    )
    return result == "INSERT 0 1"


async def upsert_neo4j(session, mapped: MappedCase) -> None:
    r = mapped.case_row
    await session.run(
        """
        MERGE (n:PotOfGreedCase {chunk_id: $chunk_id})
        SET n.case_no = $case_no,
            n.court = $court,
            n.decided_at = $decided_at,
            n.is_en_banc = $is_en_banc,
            n.validity_flag = $validity_flag
        """,
        chunk_id=r.chunk_id, case_no=r.case_no, court=r.court,
        decided_at=str(r.decided_at), is_en_banc=r.is_en_banc,
        validity_flag=r.validity_flag,
    )

    for case_id, art_id in mapped.cites:
        await session.run(
            """
            MATCH (c:PotOfGreedCase {chunk_id: $cid})
            MATCH (a:PotOfGreedArticle {chunk_id: $aid})
            MERGE (c)-[:CITES]->(a)
            """,
            cid=case_id, aid=art_id,
        )

    for case_id, art_id in mapped.based_on:
        await session.run(
            """
            MATCH (c:PotOfGreedCase {chunk_id: $cid})
            MATCH (a:PotOfGreedArticle {chunk_id: $aid})
            MERGE (c)-[:BASED_ON]->(a)
            """,
            cid=case_id, aid=art_id,
        )

    # OVERRULED_BY: 구판례가 이미 Neo4j에 있을 때만 생성
    for old_id, new_id in mapped.overruled_by:
        await session.run(
            """
            MATCH (old:PotOfGreedCase {chunk_id: $old_id})
            MATCH (new:PotOfGreedCase {chunk_id: $new_id})
            MERGE (old)-[:OVERRULED_BY]->(new)
            """,
            old_id=old_id, new_id=new_id,
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
        known_article_ids = await load_known_article_ids(pg_conn)
        print(f"적재된 article_chunks: {len(known_article_ids)}건 (스코프 기준)")

        seen_case_ids: set[str] = set()
        total_inserted = 0
        total_skipped = 0

        async with driver.session() as neo4j_session:
            for query in SEARCH_QUERIES:
                print(f"\n[검색: {query}] 판례 목록 조회 중...")
                items = await list_cases(query, max_pages=MAX_PAGES_PER_QUERY)
                print(f"  검색 결과: {len(items)}건")

                for item in items:
                    if item.case_id in seen_case_ids:
                        continue
                    seen_case_ids.add(item.case_id)

                    try:
                        raw = await fetch_case(item.case_id)
                    except Exception as exc:
                        print(f"  [SKIP] {item.case_no} 조회 오류: {exc}")
                        total_skipped += 1
                        continue

                    mapped = map_case(raw, known_article_ids)
                    if mapped is None:
                        total_skipped += 1
                        continue

                    inserted = await upsert_pg(pg_conn, mapped)
                    await upsert_neo4j(neo4j_session, mapped)
                    if inserted:
                        total_inserted += 1

        case_count = await pg_conn.fetchval("SELECT count(*) FROM case_chunks")
        print(
            f"\n완료. 신규 {total_inserted}건 적재, {total_skipped}건 스킵. "
            f"case_chunks 전체: {case_count}건"
        )
        print("validity_flag 갱신: python scripts/update_validity.py")

    finally:
        await pg_conn.close()
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
