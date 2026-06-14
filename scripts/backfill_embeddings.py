"""
샘플 데이터 embedding 백필 스크립트.

embedding이 NULL인 article_chunks / case_chunks 행에 대해
mac-server Ollama qwen3-embedding:8b 를 호출해 채운다.

사용:
    python scripts/backfill_embeddings.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings
from app.retrieval.embedder import embed_batch


async def backfill(dsn: str) -> None:
    conn = await asyncpg.connect(dsn=dsn)
    await register_vector(conn)
    try:
        for table in ("article_chunks", "case_chunks"):
            rows = await conn.fetch(
                f"SELECT chunk_id, text FROM {table} WHERE embedding IS NULL"
            )
            if not rows:
                print(f"{table}: nothing to backfill")
                continue

            chunk_ids = [r["chunk_id"] for r in rows]
            texts = [r["text"] for r in rows]
            print(f"{table}: embedding {len(texts)} rows...")
            embeddings = await embed_batch(texts)

            async with conn.transaction():
                for chunk_id, emb in zip(chunk_ids, embeddings):
                    await conn.execute(
                        f"UPDATE {table} SET embedding = $1 WHERE chunk_id = $2",
                        emb, chunk_id,
                    )
            print(f"{table}: done")
    finally:
        await conn.close()


async def main() -> None:
    settings = get_settings()
    await backfill(settings.pg_dsn)


if __name__ == "__main__":
    asyncio.run(main())
