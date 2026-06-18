from __future__ import annotations

from typing import Literal

Mode = Literal["simple", "complex"]


def classify(query: str, user_hint: Mode) -> Mode:
    """A2 수동 토글 패스스루. 결정 A: A1/A3 자동분류는 이 시그니처로 교체 가능."""
    return user_hint


def should_promote(top_score: float, threshold: float) -> bool:
    """단순 모드 top 점수 < threshold 이면 복잡 모드로 자동 승격. §5.3."""
    return top_score < threshold
