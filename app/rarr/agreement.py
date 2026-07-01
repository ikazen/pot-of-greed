from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.llm import get_llm_provider
from app.rarr.claims import parse_ref
from app.rarr.types import Claim, Evidence

_AGREEMENT_SYSTEM = (
    "주어진 주장과 근거 문서를 보고 주장이 근거와 일치하는지 판단하세요. "
    "JSON으로만 응답하세요:\n"
    "{\"agree\": true/false, \"supporting_ids\": [\"chunk_id\", ...], \"reason\": \"판단 이유\"}\n"
    "agree=true: 근거가 주장을 지지. agree=false: 근거와 불일치하거나 근거가 없음."
)


@dataclass
class AgreementResult:
    agree: bool
    supporting: list[Evidence] = field(default_factory=list)
    reason: str = ""


def _citations_grounded(cited_refs: list[str], supporting: list[Evidence]) -> bool:
    """인용된 ref가 모두 지지 근거의 ref에 포함되는가 (조 단위 정규화).

    법명은 정확하지만 무관한 근거를 인용하는 오귀속을 잡는다. 존재 자체의
    검증(할루시네이션 여부)은 verify_citations(C1) 소관이라 여기선 다루지 않고,
    파싱 불가 ref는 집합에서 제외한다. cited_refs가 없으면 trivially True.
    """
    cited = {c for c in (parse_ref(r) for r in cited_refs) if c is not None}
    if not cited:
        return True
    supported = {c for c in (parse_ref(e.ref) for e in supporting) if c is not None}
    return cited <= supported


async def check_agreement(claim: Claim, evidence: list[Evidence]) -> AgreementResult:
    """주장과 근거의 일치 여부 판정.

    grounding_check.check_claim seam을 승격·확장: agree 여부 + 지지 근거 목록 반환.
    근거 전무 또는 LLM 실패 시 보수적 폴백(agree=False).
    """
    if not evidence:
        return AgreementResult(agree=False, reason="근거 없음")

    evidence_text = "\n".join(
        f"[{e.chunk_id}] {e.ref}\n{e.text[:400]}" for e in evidence
    )
    user_msg = f"주장: {claim.text}\n\n근거:\n{evidence_text}"

    provider = get_llm_provider("aux")
    try:
        raw = await provider.chat(
            [{"role": "user", "content": user_msg}],
            system=_AGREEMENT_SYSTEM,
            json_mode=True,
            timeout=15.0,
        )
        data = json.loads(raw.strip())
        llm_agree = bool(data.get("agree", False))
        supporting_ids = set(data.get("supporting_ids", []))
        supporting = [e for e in evidence if e.chunk_id in supporting_ids]
        agree = llm_agree and _citations_grounded(claim.cited_refs, supporting)
        return AgreementResult(agree=agree, supporting=supporting, reason=data.get("reason", ""))
    except Exception:
        return AgreementResult(agree=False, reason="판정 실패 — 원문 유지")
