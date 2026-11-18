"""S5-2 지표 계산 테스트."""

import json

import pytest

from eval.dataset import Sample
from eval.metrics import (
    ALL,
    BinaryMetrics,
    binary_metrics,
    compute_metrics,
    rate,
    wilson_interval,
)

_n = 0


def s(label, lang="ko", category=None, evasion=None):
    global _n
    _n += 1
    category = category or ("jailbreak" if label else "benign")
    return Sample(
        id=f"s{_n}",
        text="x",
        label=label,
        lang=lang,
        category=category,
        source="t",
        group=f"g{_n}",
        evasion=evasion,
    )


# ------------------------------------------------------------ Wilson / Rate


@pytest.mark.parametrize(
    "k, n, lo, hi",
    [
        (0, 10, 0.0, 0.2775),
        (5, 10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0),
        (1, 8, 0.0224, 0.4709),
    ],
)
def test_wilson_known_values(k, n, lo, hi):
    got_lo, got_hi = wilson_interval(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-4)
    assert got_hi == pytest.approx(hi, abs=1e-4)


def test_wilson_edge_cases():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_rate_value():
    assert rate(1, 4).value == 0.25
    r0 = rate(0, 0)
    assert r0.value is None and (r0.ci_low, r0.ci_high) == (0.0, 1.0)


# ------------------------------------------------------------ P/R/F1


def test_binary_metrics_counts_and_scores():
    m = binary_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (m.tp, m.fn, m.fp, m.tn, m.n) == (2, 1, 1, 2, 6)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_binary_metrics_zero_division():
    m = BinaryMetrics(tp=0, fp=0, tn=5, fn=0)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_binary_metrics_length_mismatch():
    with pytest.raises(ValueError):
        binary_metrics([1, 0], [1])


# ------------------------------------------------------------ compute_metrics


@pytest.fixture
def case():
    """(샘플, 점수) — 임계값 0.5 기준 판정이 주석과 같도록 구성."""
    rows = [
        # KO 원문: 공격 3 (2 탐지), 정상 2 (1 오탐), 하드 네거티브 2 (1 오탐)
        (s(1), 0.9),
        (s(1), 0.5),  # 경계값: >= 이므로 탐지
        (s(1), 0.1),
        (s(0), 0.7),
        (s(0), 0.0),
        (s(0, category="hard_negative"), 0.6),
        (s(0, category="hard_negative"), 0.2),
        # KO 우회 변형: base64 2 (1 탐지), jamo 1 (0 탐지)
        (s(1, evasion="base64"), 0.8),
        (s(1, evasion="base64"), 0.3),
        (s(1, evasion="jamo"), 0.0),
        # EN 원문: 공격 1 (탐지), 정상 1 (통과)
        (s(1, lang="en"), 1.0),
        (s(0, lang="en"), 0.4),
        # EN 우회 변형: base64 1 (탐지)
        (s(1, lang="en", evasion="base64"), 0.95),
    ]
    return [r[0] for r in rows], [r[1] for r in rows]


def test_by_lang_uses_originals_only(case):
    m = compute_metrics(*case, threshold=0.5)
    ko = m.by_lang["ko"]
    # 정상 2 + 하드 네거티브 2 = 음성 4, 우회 변형 3건은 제외
    assert (ko.tp, ko.fn, ko.fp, ko.tn) == (2, 1, 2, 2)
    assert ko.precision == pytest.approx(0.5)
    assert ko.recall == pytest.approx(2 / 3)
    en = m.by_lang["en"]
    assert (en.tp, en.fn, en.fp, en.tn) == (1, 0, 0, 1)
    total = m.by_lang[ALL]
    assert (total.tp, total.fn, total.fp, total.tn) == (3, 1, 2, 3)
    assert list(m.by_lang) == ["en", "ko", ALL]
    assert m.n == len(case[0])


def test_false_positive_split_by_set(case):
    fp = compute_metrics(*case, threshold=0.5).false_positive
    assert (fp["ko"]["benign"].k, fp["ko"]["benign"].n) == (1, 2)
    assert (fp["ko"]["hard_negative"].k, fp["ko"]["hard_negative"].n) == (1, 2)
    assert (fp["en"]["benign"].k, fp["en"]["benign"].n) == (0, 1)
    assert fp["en"]["hard_negative"].n == 0 and fp["en"]["hard_negative"].value is None
    assert (fp[ALL]["benign"].k, fp[ALL]["benign"].n) == (1, 3)


def test_evasion_rates_per_technique(case):
    ev = compute_metrics(*case, threshold=0.5).evasion
    assert (ev["ko"]["base64"].k, ev["ko"]["base64"].n) == (1, 2)
    assert (ev["ko"]["jamo"].k, ev["ko"]["jamo"].n) == (0, 1)
    assert (ev["ko"][ALL].k, ev["ko"][ALL].n) == (1, 3)
    # EN 에는 jamo 변형이 없어도 키는 있고 n=0
    assert ev["en"]["jamo"].n == 0
    assert (ev[ALL]["base64"].k, ev[ALL]["base64"].n) == (2, 3)
    assert list(ev[ALL]) == ["base64", "jamo", ALL]


def test_threshold_changes_predictions(case):
    strict = compute_metrics(*case, threshold=0.95)
    assert strict.by_lang[ALL].tp == 1  # 1.0 만 탐지
    assert strict.false_positive[ALL]["benign"].k == 0


def test_length_mismatch(case):
    samples, scores = case
    with pytest.raises(ValueError):
        compute_metrics(samples, scores[:-1], threshold=0.5)


def test_to_dict_is_json_serializable(case):
    d = compute_metrics(*case, threshold=0.5).to_dict()
    back = json.loads(json.dumps(d))
    assert back["by_lang"]["ko"]["f1"] == pytest.approx(d["by_lang"]["ko"]["f1"])
    assert back["false_positive"]["ko"]["hard_negative"] == {
        "k": 1,
        "n": 2,
        "value": 0.5,
        "ci_low": pytest.approx(0.0945, abs=1e-4),
        "ci_high": pytest.approx(0.9055, abs=1e-4),
    }
