from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.llm import get_llm_provider
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
        agree = bool(data.get("agree", False))
        supporting_ids = set(data.get("supporting_ids", []))
        supporting = [e for e in evidence if e.chunk_id in supporting_ids]
        return AgreementResult(agree=agree, supporting=supporting, reason=data.get("reason", ""))
    except Exception:
        return AgreementResult(agree=False, reason="판정 실패 — 원문 유지")
