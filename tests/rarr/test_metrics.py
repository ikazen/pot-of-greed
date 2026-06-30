from __future__ import annotations

import pytest

from app.rarr.metrics import RarrMetrics, compute_metrics
from app.rarr.types import AttributionReport, Claim, Evidence


def _make_claim(text: str) -> Claim:
    return Claim(text=text)


def _make_evidence() -> Evidence:
    return Evidence(chunk_id="c1", ref="소득세법 제89조", text="내용", score=0.9, meta={})


def _report(
    claim_text: str,
    revised_text: str = "",
    evidence: list | None = None,
    hallucinated_refs: list | None = None,
    corrected: bool = False,
) -> AttributionReport:
    return AttributionReport(
        claim=_make_claim(claim_text),
        revised_text=revised_text or claim_text,
        evidence=evidence or [],
        hallucinated_refs=hallucinated_refs or [],
        corrected=corrected,
    )


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m.n_claims == 0
    assert m.attribution_score == 1.0
    assert m.preservation_score == 1.0
    assert m.n_hallucinated == 0
    assert m.hallucination_correction_rate == 1.0


def test_compute_metrics_all_attributed():
    reports = [
        _report("주장1", evidence=[_make_evidence()]),
        _report("주장2", evidence=[_make_evidence()]),
    ]
    m = compute_metrics(reports)
    assert m.n_claims == 2
    assert m.attribution_score == 1.0


def test_compute_metrics_partial_attribution():
    reports = [
        _report("주장1", evidence=[_make_evidence()]),
        _report("주장2", evidence=[]),
    ]
    m = compute_metrics(reports)
    assert m.attribution_score == pytest.approx(0.5)


def test_compute_metrics_preservation_identical():
    reports = [_report("동일한 주장", revised_text="동일한 주장")]
    m = compute_metrics(reports)
    assert m.preservation_score == pytest.approx(1.0)


def test_compute_metrics_preservation_changed():
    reports = [_report("원래 주장 내용", revised_text="완전히 다른 내용입니다")]
    m = compute_metrics(reports)
    assert 0.0 <= m.preservation_score < 1.0


def test_compute_metrics_preservation_strips_suffix_tags():
    reports = [
        _report("주장 A", revised_text="주장 A [미검증]"),
        _report("주장 B", revised_text="주장 B [정정: X → Y]"),
    ]
    m = compute_metrics(reports)
    assert m.preservation_score == pytest.approx(1.0)


def test_compute_metrics_no_hallucinations():
    reports = [_report("주장")]
    m = compute_metrics(reports)
    assert m.n_hallucinated == 0
    assert m.hallucination_correction_rate == 1.0


def test_compute_metrics_hallucination_all_corrected():
    reports = [
        _report("주장", hallucinated_refs=["소득세법 제999조"], corrected=True),
        _report("주장2", hallucinated_refs=["법인세법 제1000조"], corrected=True),
    ]
    m = compute_metrics(reports)
    assert m.n_hallucinated == 2
    assert m.hallucination_correction_rate == pytest.approx(1.0)


def test_compute_metrics_hallucination_partial_correction():
    reports = [
        _report("주장1", hallucinated_refs=["가짜조문"], corrected=True),
        _report("주장2", hallucinated_refs=["또다른가짜"], corrected=False),
    ]
    m = compute_metrics(reports)
    assert m.n_hallucinated == 2
    assert m.hallucination_correction_rate == pytest.approx(0.5)


def test_compute_metrics_hallucination_none_corrected():
    reports = [
        _report("주장", hallucinated_refs=["가짜조문"], corrected=False),
    ]
    m = compute_metrics(reports)
    assert m.n_hallucinated == 1
    assert m.hallucination_correction_rate == pytest.approx(0.0)


def test_compute_metrics_returns_rarr_metrics_type():
    reports = [_report("주장", evidence=[_make_evidence()])]
    m = compute_metrics(reports)
    assert isinstance(m, RarrMetrics)
