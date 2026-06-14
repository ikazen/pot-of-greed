from __future__ import annotations

from dataclasses import dataclass

from app.db.pg import get_pool


@dataclass
class Chunk:
    chunk_id: str
    table: str        # "article" | "case"
    text: str
    score: float
    meta: dict


async def vector_search(
    embedding: list[float],
    top_k: int = 30,
    only_current: bool = True,
) -> list[Chunk]:
    pool = get_pool()
    current_filter = "AND is_current = TRUE" if only_current else ""

    article_sql = f"""
        SELECT chunk_id, text, law_name, article_no, clause_path, is_current,
               1 - (embedding <=> $1::vector) AS score
        FROM article_chunks
        WHERE embedding IS NOT NULL {current_filter}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """
    case_sql = f"""
        SELECT chunk_id, text, case_no, court, validity_flag, decided_at,
               1 - (embedding <=> $1::vector) AS score
        FROM case_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """
    vec = embedding
    async with pool.acquire() as conn:
        art_rows = await conn.fetch(article_sql, vec, top_k)
        case_rows = await conn.fetch(case_sql, vec, top_k)

    results: list[Chunk] = []
    for row in art_rows:
        results.append(Chunk(
            chunk_id=row["chunk_id"],
            table="article",
            text=row["text"],
            score=float(row["score"]),
            meta={
                "law_name": row["law_name"],
                "article_no": row["article_no"],
                "clause_path": row["clause_path"],
                "is_current": row["is_current"],
            },
        ))
    for row in case_rows:
        results.append(Chunk(
            chunk_id=row["chunk_id"],
            table="case",
            text=row["text"],
            score=float(row["score"]),
            meta={
                "case_no": row["case_no"],
                "court": row["court"],
                "validity_flag": row["validity_flag"],
                "decided_at": str(row["decided_at"]),
            },
        ))
    return sorted(results, key=lambda c: c.score, reverse=True)[:top_k]
