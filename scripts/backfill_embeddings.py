"""
임베딩 백필 스크립트 — article_chunks + case_chunks NULL 행을 배치로 처리.

사용:
    python scripts/backfill_embeddings.py
    python scripts/backfill_embeddings.py --batch-size 32 --concurrency 1
    python scripts/backfill_embeddings.py --rebuild-index   # 대량 적재 후 hnsw 일괄 빌드

옵션:
    --batch-size N     한 번에 embed_batch에 보낼 행 수 (기본 64)
    --concurrency N    동시 Ollama embed_batch 호출 수 (기본 2, mac-server 보호)
    --rebuild-index    백필 전 hnsw 인덱스 DROP → 백필 후 일괄 CREATE
                       (대량 적재 시 점진 삽입보다 빠를 수 있음, 기본 off)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings
from app.retrieval.embedder import embed_batch

_HNSW_INDEXES = {
    "article_chunks": "article_chunks_embedding_idx",
    "case_chunks": "case_chunks_embedding_idx",
}
_HNSW_DEF = {
    "article_chunks": "CREATE INDEX article_chunks_embedding_idx ON article_chunks USING hnsw (embedding vector_cosine_ops)",
    "case_chunks": "CREATE INDEX case_chunks_embedding_idx ON case_chunks USING hnsw (embedding vector_cosine_ops)",
}


async def _drop_hnsw(conn: asyncpg.Connection, table: str) -> None:
    idx = _HNSW_INDEXES[table]
    await conn.execute(f"DROP INDEX IF EXISTS {idx}")
    print(f"{table}: hnsw 인덱스 DROP ({idx})")


async def _create_hnsw(conn: asyncpg.Connection, table: str) -> None:
    print(f"{table}: hnsw 인덱스 빌드 중... (시간 소요)")
    await conn.execute(_HNSW_DEF[table])
    print(f"{table}: hnsw 인덱스 빌드 완료")


async def backfill_table(
    conn: asyncpg.Connection,
    table: str,
    batch_size: int,
    sem: asyncio.Semaphore,
) -> None:
    total: int = await conn.fetchval(
        f"SELECT count(*) FROM {table} WHERE embedding IS NULL"
    )
    if total == 0:
        print(f"{table}: nothing to backfill")
        return

    print(f"{table}: {total}행 백필 시작 (batch={batch_size})")
    done = 0

    while True:
        # embedding IS NULL 행을 batch_size만큼 반복 조회 (커밋 후 사라지므로 OFFSET 불필요)
        rows = await conn.fetch(
            f"SELECT chunk_id, text FROM {table} WHERE embedding IS NULL LIMIT $1",
            batch_size,
        )
        if not rows:
            break

        chunk_ids = [r["chunk_id"] for r in rows]
        texts = [r["text"] for r in rows]

        async with sem:
            embeddings = await embed_batch(texts)

        async with conn.transaction():
            for chunk_id, emb in zip(chunk_ids, embeddings):
                await conn.execute(
                    f"UPDATE {table} SET embedding = $1 WHERE chunk_id = $2",
                    emb, chunk_id,
                )

        done += len(rows)
        print(f"{table}: {done}/{total}")

    remaining = await conn.fetchval(
        f"SELECT count(*) FROM {table} WHERE embedding IS NULL"
    )
    print(f"{table}: 완료 (미처리 잔여={remaining})")


async def main(batch_size: int, concurrency: int, rebuild_index: bool) -> None:
    settings = get_settings()
    conn = await asyncpg.connect(dsn=settings.pg_dsn)
    await register_vector(conn)
    sem = asyncio.Semaphore(concurrency)

    try:
        for table in ("article_chunks", "case_chunks"):
            if rebuild_index:
                await _drop_hnsw(conn, table)

            await backfill_table(conn, table, batch_size, sem)

            if rebuild_index:
                await _create_hnsw(conn, table)

        # 최종 검증
        for table in ("article_chunks", "case_chunks"):
            null_cnt = await conn.fetchval(
                f"SELECT count(*) FROM {table} WHERE embedding IS NULL"
            )
            total_cnt = await conn.fetchval(f"SELECT count(*) FROM {table}")
            print(f"{table}: 전체 {total_cnt}행, embedding NULL {null_cnt}행")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임베딩 백필")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="백필 전 hnsw DROP → 완료 후 일괄 CREATE (대량 적재 시 권장)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.batch_size, args.concurrency, args.rebuild_index))
