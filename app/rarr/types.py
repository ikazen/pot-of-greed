from __future__ import annotations

from dataclasses import dataclass, field


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
    revised_text: str = ""
    corrections: list[str] = field(default_factory=list)
    hallucinated_refs: list[str] = field(default_factory=list)
    corrected: bool = False
