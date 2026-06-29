from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.llm import get_llm_provider
from app.retrieval.vector_search import Chunk

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    grounded: bool
    issues: list[str] = field(default_factory=list)


_GROUNDING_SYSTEM = (
    "당신은 세법 답변의 근거 검증자입니다. "
    "제공된 검색 근거 청크들과 대조하여 답변의 각 주장이 근거에 실재하는지 확인하십시오. "
    "근거가 있으면: {\"grounded\": true, \"issues\": []} "
    "근거 없는 주장이 있으면: {\"grounded\": false, \"issues\": [\"주장1\", \"주장2\"]} "
    "다른 텍스트는 출력하지 마십시오."
)


async def check_answer(answer: str, sources: list[Chunk]) -> GroundingResult:
    """F1: 답변 전체를 검색 근거와 교차확인.

    LLM 호출 실패 시 grounded=True 폴백 (게이트 비활성화보다 서비스 유지 우선).
    """
    context_preview = "\n".join(
        f"[{c.chunk_id}] {c.text[:300]}" for c in sources[:8]
    )
    user_msg = f"답변:\n{answer}\n\n검색 근거:\n{context_preview}"

    try:
        provider = get_llm_provider()
        raw = await provider.chat(
            [{"role": "user", "content": user_msg}],
            system=_GROUNDING_SYSTEM,
            json_mode=True,
            timeout=10.0,
        )
        data = json.loads(raw.strip())
        return GroundingResult(
            grounded=bool(data.get("grounded", True)),
            issues=list(data.get("issues", [])),
        )
    except Exception:
        logger.debug("grounding check fallback to grounded=True", exc_info=True)
        return GroundingResult(grounded=True)


async def check_claim(claim: str, sources: list[Chunk]) -> bool:
    """F2 확장 seam (시그니처 고정 — 결정 F). 현재는 check_answer 위임."""
    result = await check_answer(claim, sources)
    return result.grounded


def apply_grounding(
    raw_answer: str,
    result: GroundingResult,
    action: str,
) -> str:
    """grounding_action 설정에 따라 답변에 경고 플래그 부착 또는 미검증 주장 제거.

    action="flag": 답변 앞에 [주의] 경고 추가.
    action="strip": issues 목록에 있는 주장 문장 제거.
    """
    if result.grounded or not result.issues:
        return raw_answer

    if action == "strip":
        lines = raw_answer.split("\n")
        filtered = [
            line for line in lines
            if not any(issue[:20] in line for issue in result.issues)
        ]
        return "\n".join(filtered)

    # default: flag
    issues_text = "\n".join(f"- {i}" for i in result.issues)
    warning = f"[주의] 다음 주장은 검색 근거에서 확인되지 않았습니다:\n{issues_text}\n\n"
    return warning + raw_answer
