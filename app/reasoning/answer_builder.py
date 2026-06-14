from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.vector_search import Chunk

_WARNING_MESSAGES = {
    "overruled": "이 판례는 이후 판례에 의해 변경되었습니다. 현행 법리 적용 시 결론이 달라질 수 있습니다.",
    "law_amended": "이 판례의 근거 조문이 판결 이후 개정되었습니다. 현행법 적용 시 결론이 달라질 수 있습니다.",
    "uncertain": "이 판례의 현행 유효성이 불확실합니다. 최신 판례를 별도로 확인하십시오.",
}


@dataclass
class Source:
    type: str        # "article" | "case"
    ref: str
    chunk_id: str


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


def build_answer(raw_answer: str, chunks: list[Chunk]) -> Answer:
    sources: list[Source] = []
    warnings: list[Warning] = []

    for chunk in chunks:
        if chunk.table == "article":
            ref = (
                f"{chunk.meta.get('law_name', '')} "
                f"{chunk.meta.get('article_no', '')} "
                f"{chunk.meta.get('clause_path', '') or ''}"
            ).strip()
            sources.append(Source(type="article", ref=ref, chunk_id=chunk.chunk_id))
        else:
            ref = chunk.meta.get("case_no", chunk.chunk_id)
            sources.append(Source(type="case", ref=ref, chunk_id=chunk.chunk_id))
            flag = chunk.meta.get("validity_flag")
            if flag and flag in _WARNING_MESSAGES:
                warnings.append(Warning(
                    chunk_id=chunk.chunk_id,
                    ref=ref,
                    validity_flag=flag,
                    message=_WARNING_MESSAGES[flag],
                ))

    return Answer(answer=raw_answer, sources=sources, warnings=warnings)
