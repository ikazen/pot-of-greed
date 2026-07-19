from __future__ import annotations

from app.rarr.debug import build_debug_trace
from app.rarr.types import AttributionReport, Claim, Evidence


def _report(text, revised=None, agree=True, reason="", removed_refs=None):
    return AttributionReport(
        claim=Claim(text=text),
        evidence=[Evidence(chunk_id="c1", ref="소득세법 제89조", text="본문", score=0.9, meta={})],
        agree=agree,
        agreement_reason=reason,
        revised_text=revised if revised is not None else text,
        removed_refs=removed_refs or [],
    )


def test_build_debug_trace_unchanged_claim_has_null_revised():
    reports = [_report("원문 그대로")]
    trace = build_debug_trace(reports, mode="simple", deferred_count=0, hallucinated_all=set(), legal_reasoning_applied=False)
    assert trace["claims"][0]["revised"] is None
    assert trace["claims"][0]["original"] == "원문 그대로"


def test_build_debug_trace_changed_claim_has_revised_text():
    reports = [_report("원문", revised="수정된 문장", agree=False, reason="근거 불일치")]
    trace = build_debug_trace(reports, mode="simple", deferred_count=0, hallucinated_all=set(), legal_reasoning_applied=False)
    claim = trace["claims"][0]
    assert claim["revised"] == "수정된 문장"
    assert claim["agree"] is False
    assert claim["agreement_reason"] == "근거 불일치"


def test_build_debug_trace_includes_scrubbed_refs():
    reports = [_report("소득세법 제999조 언급", removed_refs=["소득세법 제999조"])]
    trace = build_debug_trace(
        reports, mode="complex", deferred_count=1,
        hallucinated_all={"소득세법 제999조"}, legal_reasoning_applied=True,
    )
    assert trace["scrubbed_refs"] == ["소득세법 제999조"]
    assert trace["deferred_count"] == 1
    assert trace["legal_reasoning_applied"] is True
    assert trace["claims"][0]["removed_refs"] == ["소득세법 제999조"]


def test_build_debug_trace_evidence_refs_and_count():
    reports = [_report("문장")]
    trace = build_debug_trace(reports, mode="simple", deferred_count=0, hallucinated_all=set(), legal_reasoning_applied=False)
    claim = trace["claims"][0]
    assert claim["evidence_refs"] == ["소득세법 제89조"]
    assert claim["evidence_count"] == 1
