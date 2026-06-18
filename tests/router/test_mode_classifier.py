from __future__ import annotations

import pytest
from app.router.mode_classifier import classify, should_promote


def test_classify_simple_passthrough():
    assert classify("부가가치세 신고 기한은?", "simple") == "simple"


def test_classify_complex_passthrough():
    assert classify("법인세법 제52조와 소득세법 제14조의 관계는?", "complex") == "complex"


def test_should_promote_below_threshold():
    assert should_promote(0.3, 0.5) is True


def test_should_promote_above_threshold():
    assert should_promote(0.7, 0.5) is False


def test_should_promote_at_threshold():
    # 경계값: threshold 이상이면 승격 안 함
    assert should_promote(0.5, 0.5) is False


def test_should_promote_zero_score():
    assert should_promote(0.0, 0.5) is True
