"""
validity_flag 갱신 스크립트 (1층 기계적 판정).

Neo4j 그래프를 읽어 PG case_chunks.validity_flag 를 계산·업데이트한다.
데이터 변경(새 판례/개정 추가) 시 재실행.

판정 규칙:
  1. OVERRULED_BY 엣지가 있는 케이스 → overruled
  2. BASED_ON 조문이 판결일 이후에 AMENDED_BY를 가지면 → law_amended
  3. 그 외 → valid

사용:
    python scripts/update_validity.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from neo4j import AsyncGraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings


async def compute_flags(neo4j_uri: str, user: str, password: str) -> dict[str, str]:
    """chunk_id → validity_flag 딕셔너리 반환."""
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(user, password))
    flags: dict[str, str] = {}
    try:
        async with driver.session() as session:
            # 1. OVERRULED_BY 있는 케이스
            r1 = await session.run(
                """
                MATCH (c:PotOfGreedCase)-[:OVERRULED_BY]->()
                RETURN c.chunk_id AS chunk_id
                """
            )
            async for record in r1:
                flags[record["chunk_id"]] = "overruled"

            # 2. BASED_ON 조문이 판결일 이후 AMENDED_BY를 가지는 케이스
            r2 = await session.run(
                """
                MATCH (c:PotOfGreedCase)-[:BASED_ON]->(a:PotOfGreedArticle)
                      -[:AMENDED_BY]->(m:PotOfGreedAmendment)
                WHERE m.amended_at > c.decided_at
                  AND NOT (c)-[:OVERRULED_BY]->()
                RETURN DISTINCT c.chunk_id AS chunk_id
                """
            )
            async for record in r2:
                cid = record["chunk_id"]
                if cid not in flags:
                    flags[cid] = "law_amended"

            # 3. 나머지 PotOfGreedCase → valid
            r3 = await session.run(
                "MATCH (c:PotOfGreedCase) RETURN c.chunk_id AS chunk_id"
            )
            async for record in r3:
                cid = record["chunk_id"]
                if cid not in flags:
                    flags[cid] = "valid"
    finally:
        await driver.close()
    return flags


async def apply_flags(dsn: str, flags: dict[str, str]) -> None:
    conn = await asyncpg.connect(dsn=dsn)
    try:
        async with conn.transaction():
            for chunk_id, flag in flags.items():
                await conn.execute(
                    "UPDATE case_chunks SET validity_flag = $1 WHERE chunk_id = $2",
                    flag, chunk_id,
                )
        print(f"Updated {len(flags)} case validity_flag(s).")
        for cid, flag in sorted(flags.items()):
            print(f"  {cid}: {flag}")
    finally:
        await conn.close()


async def main() -> None:
    settings = get_settings()
    print("Computing validity flags from Neo4j graph...")
    flags = await compute_flags(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    print(f"Applying {len(flags)} flag(s) to PG...")
    await apply_flags(settings.pg_dsn, flags)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
