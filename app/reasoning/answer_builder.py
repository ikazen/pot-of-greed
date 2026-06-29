from __future__ import annotations

import logging
from dataclasses import dataclass

from app.retrieval.vector_search import Chunk

logger = logging.getLogger(__name__)


def _build_warning_message(flag: str, meta: dict) -> str:
    """§4 표기 방식: 판례별 구체적 경고 문구 생성."""
    if flag == "overruled":
        return "[주의] 이 판례는 이후 판례에 의해 변경되었습니다. 현행 법리 적용 시 결론이 달라질 수 있습니다."
    if flag == "law_amended":
        article = meta.get("amended_article") or meta.get("article_ref") or ""
        suffix = f" ({article} 개정)" if article else ""
        return f"[주의] 이 판례의 근거 조문이 판결 이후 개정되었습니다{suffix}. 현행법 적용 시 결론이 달라질 수 있습니다."
    if flag == "uncertain":
        return "[주의] 이 판례의 현행 유효성이 불확실합니다. 최신 판례를 별도로 확인하십시오."
    return f"[주의] 유효성 상태: {flag}"


@dataclass
class Source:
    type: str        # "article" | "case"
    ref: str
    chunk_id: str
    summary: str


@dataclass
class Warning:
    chunk_id: str
    ref: str
    validity_flag: str
    message: str


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    warnings: list[Warning]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [vars(s) for s in self.sources],
            "warnings": [vars(w) for w in self.warnings],
        }


_VALIDITY_FLAGS = {"overruled", "law_amended", "uncertain"}


def _first_line(text: str, limit: int = 100) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first[:limit] + ("..." if len(first) > limit else "")


def _is_cited_article(raw_answer: str, meta: dict) -> bool:
    law_name = meta.get("law_name", "")
    article_no = meta.get("article_no", "")
    if not law_name or not article_no:
        return False
    return (law_name + " " + article_no) in raw_answer or (law_name + article_no) in raw_answer


def _is_cited_case(raw_answer: str, meta: dict) -> bool:
    case_no = meta.get("case_no", "")
    return bool(case_no) and case_no in raw_answer


def build_answer(raw_answer: str, chunks: list[Chunk], limit: int = 3) -> Answer:
    # 부모 컨텍스트 청크 제외 (LLM 컨텍스트 보강용 — 표시 불필요)
    candidates = [c for c in chunks if c.meta.get("context_role") != "parent"]

    # ref 기준 dedup (첫 등장 우선, rerank 순서상 score 높은 것이 먼저 위치)
    seen_refs: set[str] = set()
    deduped: list[Chunk] = []
    for c in candidates:
        if c.table == "article":
            ref = (
                f"{c.meta.get('law_name', '')} "
                f"{c.meta.get('article_no', '')} "
                f"{c.meta.get('clause_path', '') or ''}"
            ).strip()
        else:
            ref = c.meta.get("case_no", c.chunk_id)
        if ref not in seen_refs:
            seen_refs.add(ref)
            deduped.append(c)

    # 본문 인용 판정 → cited 우선, 같은 그룹 내 score 내림차순
    def _sort_key(c: Chunk) -> tuple[int, float]:
        if c.table == "article":
            cited = _is_cited_article(raw_answer, c.meta)
        else:
            cited = _is_cited_case(raw_answer, c.meta)
        return (0 if cited else 1, -c.score)

    selected = sorted(deduped, key=_sort_key)[:limit]

    sources: list[Source] = []
    warnings: list[Warning] = []

    for chunk in selected:
        if chunk.table == "article":
            ref = (
                f"{chunk.meta.get('law_name', '')} "
                f"{chunk.meta.get('article_no', '')} "
                f"{chunk.meta.get('clause_path', '') or ''}"
            ).strip()
            sources.append(Source(
                type="article",
                ref=ref,
                chunk_id=chunk.chunk_id,
                summary=_first_line(chunk.text),
            ))
        else:
            ref = chunk.meta.get("case_no", chunk.chunk_id)
            sources.append(Source(
                type="case",
                ref=ref,
                chunk_id=chunk.chunk_id,
                summary=_first_line(chunk.text),
            ))
            flag = chunk.meta.get("validity_flag")
            if flag and flag in _VALIDITY_FLAGS:
                warnings.append(Warning(
                    chunk_id=chunk.chunk_id,
                    ref=ref,
                    validity_flag=flag,
                    message=_build_warning_message(flag, chunk.meta),
                ))

    return Answer(answer=raw_answer, sources=sources, warnings=warnings)


_LEGAL_REASONING_SYSTEM = (
    "당신은 세법 법리 분석 전문가입니다. "
    "제공된 판례 유효성 정보(1층: validity_flag, 2층: 시점 정합)를 바탕으로 "
    "현재 법리가 여전히 유효한지, 개정된 조문에도 동일 법리가 적용되는지 판단하십시오. "
    "판단 근거를 2~3문장으로 간결하게 작성하십시오."
)


async def legal_reasoning_layer(
    query: str,
    chunks: list[Chunk],
    warnings: list[Warning],
) -> str | None:
    """3층 법리 판단 (복잡 모드 한정) — 1·2층 사실을 컨텍스트로 제공 후 LLM 판단.

    판단 결과 문자열 반환. 경고가 없거나 LLM 오류 시 None 반환.
    """
    if not warnings:
        return None

    validity_facts = "\n".join(
        f"- {w.ref}: {w.message}" for w in warnings
    )
    article_refs = "\n".join(
        f"- {c.meta.get('law_name', '')} {c.meta.get('article_no', '')}"
        for c in chunks if c.table == "article"
    )
    user_msg = (
        f"질의: {query}\n\n"
        f"유효성 경고 (1층):\n{validity_facts}\n\n"
        f"관련 조문 (2층):\n{article_refs or '없음'}\n\n"
        "위 사실을 바탕으로 현재 법리 유효성을 판단하십시오."
    )

    try:
        from app.llm import get_llm_provider
        provider = get_llm_provider()
        result = await provider.chat(
            [{"role": "user", "content": user_msg}],
            system=_LEGAL_REASONING_SYSTEM,
            timeout=10.0,
        )
        return result.strip() or None
    except Exception:
        logger.debug("legal_reasoning_layer failed", exc_info=True)
        return None
