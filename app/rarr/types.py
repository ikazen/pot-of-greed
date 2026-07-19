from __future__ import annotations

from dataclasses import dataclass, field

# M1: agreement/edit이 LLM에 붙이는 evidence 스니펫 길이. 400자는 긴 조문·판례의
# 지지 문구가 절단 밖에 위치해 거짓 불일치(불필요한 edit)를 유발해 800으로 상향.
EVIDENCE_SNIPPET_CHARS = 800


@dataclass
class Claim:
    text: str
    cited_refs: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    chunk_id: str
    ref: str
    text: str
    score: float
    meta: dict


@dataclass
class AttributionReport:
    claim: Claim
    evidence: list[Evidence] = field(default_factory=list)
    agree: bool = True
    agreement_reason: str = ""
    revised_text: str = ""
    corrections: list[str] = field(default_factory=list)
    hallucinated_refs: list[str] = field(default_factory=list)
    corrected: bool = False
    removed_refs: list[str] = field(default_factory=list)
