from __future__ import annotations

import json
import re
import time

from app.llm import get_llm_provider
from app.rarr.types import Claim

# 조문 번호: "XXX법 제N조" 형식
_RE_ARTICLE = re.compile(r"(?P<law>[\w가-힣]+법)\s*(?P<article>제\d+조)(?:\s*제\d+항)?")
# 판례 번호: 연도+사건부호+번호 (예: 2018두12345, 2021도1234)
_RE_CASE = re.compile(r"\d{4}[가-힣]+\d+")

_DECOMPOSE_SYSTEM = (
    "주어진 세법 답변을 독립적인(decontextualized) 원자 주장(atomic claim) 목록으로 분해하세요. "
    "각 주장은 다른 주장 없이도 의미가 통해야 합니다. "
    "JSON 배열만 반환하세요: [{\"text\": \"주장 내용\"}, ...]\n"
    "예시: [{\"text\": \"1세대1주택 비과세는 보유기간 2년 이상이 필요하다.\"}, ...]"
)


_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """구두점/개행 기준 문장 분리 (rule-based, 폴백 전용).

    decompose LLM이 실패했을 때 draft 전체를 단일 claim으로 뭉치지 않기 위한
    최소한의 다항 유지책. 형태소 수준 분리는 하지 않는다.
    """
    parts = [p.strip() for p in _RE_SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


def _extract_refs(text: str) -> list[str]:
    matches = list(_RE_ARTICLE.finditer(text)) + list(_RE_CASE.finditer(text))
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        r = m.group(0).strip()
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


def parse_ref(ref: str) -> tuple[str, tuple[str, ...]] | None:
    """ref 문자열 → 구조적 동등 매칭용 (kind, params).

    ("article", (law_name, article_no)) | ("case", (case_no,)) | None(파싱 불가).
    추출(_extract_refs)과 같은 정규식을 써서 divergence를 방지한다.
    항(제N항)은 소비만 하고 존재 판정은 조 단위로 한다.
    """
    ref = ref.strip()
    m = _RE_ARTICLE.match(ref)
    if m:
        return "article", (m.group("law"), m.group("article"))
    if _RE_CASE.fullmatch(ref):
        return "case", (ref,)
    return None


async def decompose_claims(draft_text: str, deadline: float | None = None) -> list[Claim]:
    """draft 텍스트를 원자 주장으로 분해.

    H1: deadline이 주어지면 남은 시간으로 timeout을 클램프하고, 이미 초과했으면
    LLM 호출 없이 규칙기반 폴백(_split_sentences)으로 직행한다.
    """
    timeout = 15.0
    skip_llm = False
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            skip_llm = True
        else:
            timeout = min(timeout, remaining)

    if not skip_llm:
        provider = get_llm_provider("aux")
        try:
            raw = await provider.chat(
                [{"role": "user", "content": draft_text}],
                system=_DECOMPOSE_SYSTEM,
                json_mode=True,
                timeout=timeout,
            )
            items = json.loads(raw.strip())
            claims = [
                Claim(text=item["text"], cited_refs=_extract_refs(item["text"]))
                for item in items
                if isinstance(item, dict) and item.get("text")
            ]
            if claims:
                return claims
        except Exception:
            pass

    sentences = _split_sentences(draft_text)
    if not sentences:
        return [Claim(text=draft_text, cited_refs=_extract_refs(draft_text))]
    return [Claim(text=s, cited_refs=_extract_refs(s)) for s in sentences]
