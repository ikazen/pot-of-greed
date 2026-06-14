from __future__ import annotations

from app.retrieval.vector_search import Chunk


def rrf_fuse(
    vec_results: list[Chunk],
    kw_results: list[Chunk],
    k: int = 60,
    top_n: int = 30,
) -> list[Chunk]:
    """Reciprocal Rank Fusion of two ranked lists.

    Score(d) = sum_over_lists( 1 / (k + rank(d)) )
    chunk_id로 dedup, 두 리스트에 없으면 해당 리스트 순위 = len+1.
    """
    vec_rank: dict[str, int] = {c.chunk_id: i + 1 for i, c in enumerate(vec_results)}
    kw_rank: dict[str, int] = {c.chunk_id: i + 1 for i, c in enumerate(kw_results)}

    all_chunks: dict[str, Chunk] = {}
    for c in vec_results + kw_results:
        if c.chunk_id not in all_chunks:
            all_chunks[c.chunk_id] = c

    vec_miss = len(vec_results) + 1
    kw_miss = len(kw_results) + 1

    scored: list[tuple[float, Chunk]] = []
    for chunk_id, chunk in all_chunks.items():
        rrf_score = (
            1.0 / (k + vec_rank.get(chunk_id, vec_miss))
            + 1.0 / (k + kw_rank.get(chunk_id, kw_miss))
        )
        scored.append((rrf_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]
