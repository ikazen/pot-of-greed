from __future__ import annotations

from app.llm import get_llm_provider
from app.rarr.agreement import AgreementResult
from app.rarr.types import Claim, Evidence

_EDIT_SYSTEM = (
    "당신은 세법 답변 교정 전문가입니다. "
    "주어진 주장과 근거를 보고 주장의 오류를 최소한으로 수정하세요.\n"
    "규칙:\n"
    "1. 근거와 불일치하는 부분만 수정. 나머지 문장과 문체는 그대로 보존.\n"
    "2. 잘못된 조문/판례 번호는 근거에서 찾은 실제 번호로 교체하거나 제거.\n"
    "   교정한 경우 수정된 주장 끝에 [정정: 원래 내용 → 수정 내용] 형식으로 기록.\n"
    "3. 오직 수정된 주장 텍스트만 반환. 설명 없이."
)


async def edit_claim(
    claim: Claim,
    agreement: AgreementResult,
    evidence: list[Evidence],
) -> tuple[str, list[Evidence], list[str]]:
    """주장 최소 수정. agree 주장은 건드리지 않음.

    반환: (revised_text, used_evidence, corrections)
    근거 전무 시 원문 + [미검증] 플래그.
    LLM 실패 시 원문 유지 폴백.
    """
    if agreement.agree:
        return claim.text, agreement.supporting, []

    if not evidence:
        return claim.text + " [미검증]", [], []

    evidence_text = "\n".join(
        f"[{e.chunk_id}] {e.ref}\n{e.text[:400]}" for e in evidence
    )
    user_msg = f"원래 주장: {claim.text}\n\n근거:\n{evidence_text}"

    provider = get_llm_provider("edit")
    try:
        revised = await provider.chat(
            [{"role": "user", "content": user_msg}],
            system=_EDIT_SYSTEM,
            timeout=20.0,
        )
        revised = revised.strip()
        corrections = _extract_corrections(revised)
        return revised, evidence, corrections
    except Exception:
        return claim.text, [], []


def _extract_corrections(text: str) -> list[str]:
    """[정정: ...] 패턴 추출."""
    import re
    return re.findall(r"\[정정:[^\]]+\]", text)
