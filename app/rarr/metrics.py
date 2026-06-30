from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from app.rarr.types import AttributionReport

_RE_SUFFIX = re.compile(r"\s*\[미검증\]|\s*\[정정:[^\]]*\]")


def _strip_tags(text: str) -> str:
    return _RE_SUFFIX.sub("", text).strip()


@dataclass
class RarrMetrics:
    n_claims: int
    attribution_score: float
    preservation_score: float
    n_hallucinated: int
    hallucination_correction_rate: float


def compute_metrics(reports: list[AttributionReport]) -> RarrMetrics:
    n = len(reports)
    if n == 0:
        return RarrMetrics(
            n_claims=0,
            attribution_score=1.0,
            preservation_score=1.0,
            n_hallucinated=0,
            hallucination_correction_rate=1.0,
        )

    attributed = sum(1 for r in reports if r.evidence)
    attribution_score = attributed / n

    preservation_scores = []
    for r in reports:
        a = _strip_tags(r.claim.text)
        b = _strip_tags(r.revised_text) if r.revised_text else a
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        preservation_scores.append(ratio)
    preservation_score = sum(preservation_scores) / n

    hallucinated = [r for r in reports if r.hallucinated_refs]
    n_hallucinated = len(hallucinated)
    if n_hallucinated == 0:
        hallucination_correction_rate = 1.0
    else:
        corrected = sum(1 for r in hallucinated if r.corrected)
        hallucination_correction_rate = corrected / n_hallucinated

    return RarrMetrics(
        n_claims=n,
        attribution_score=attribution_score,
        preservation_score=preservation_score,
        n_hallucinated=n_hallucinated,
        hallucination_correction_rate=hallucination_correction_rate,
    )
