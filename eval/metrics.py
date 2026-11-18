"""평가 지표 계산 (S5-2). 순수 Python (sklearn 불필요).

``compute_metrics(samples, scores, threshold)`` 가 세 묶음을 계산한다.

1. ``by_lang`` — 언어별 Precision / Recall / F1.
   **원문만**(``evasion is None``) 사용한다. test 공격의 약 3/4 이 우회 변형이라
   섞으면 언어별 지표가 사실상 우회 탐지율이 되기 때문. 우회 변형은 3번에서 따로 본다.
2. ``false_positive`` — 오탐률. 정상(``benign``)과 하드 네거티브 세트 각각.
3. ``evasion`` — 우회 공격 세트의 기법별 탐지율 (= 해당 변형 공격 중 공격으로 판정된 비율).

2·3 은 표본이 작은 칸이 많아서(test 하드 네거티브 EN 8·KO 11건) 비율만 보면 오해하기 쉽다.
그래서 비율마다 건수 ``k/n`` 과 Wilson 95% 신뢰구간을 함께 둔다.
각 묶음에는 언어별 값과 함께 전체 합계 키 ``"all"`` 이 들어간다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from eval.dataset import Sample

__all__ = [
    "ALL",
    "BinaryMetrics",
    "EvalMetrics",
    "Rate",
    "binary_metrics",
    "compute_metrics",
    "rate",
    "wilson_interval",
]

ALL = "all"
Z_95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """이항 비율 k/n 의 Wilson 신뢰구간. n=0 이면 (0, 1)."""
    if not 0 <= k <= n:
        raise ValueError(f"0 <= k <= n 이어야 합니다: k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class Rate:
    """비율 k/n 과 Wilson 95% 신뢰구간. n=0 이면 ``value`` 는 None."""

    k: int
    n: int
    value: float | None
    ci_low: float
    ci_high: float


def rate(k: int, n: int) -> Rate:
    lo, hi = wilson_interval(k, n)
    return Rate(k=k, n=n, value=(k / n if n else None), ci_low=lo, ci_high=hi)


def _div(a: int, b: int) -> float:
    # 분모 0 이면 0.0 (sklearn zero_division=0 과 동일한 관례)
    return a / b if b else 0.0


@dataclass(frozen=True)
class BinaryMetrics:
    """혼동행렬과 Precision / Recall / F1 (양성 = 공격)."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        return _div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _div(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "n": self.n,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def binary_metrics(labels: Iterable[int], preds: Iterable[bool | int]) -> BinaryMetrics:
    tp = fp = tn = fn = 0
    for y, p in zip(labels, preds, strict=True):
        if y and p:
            tp += 1
        elif y:
            fn += 1
        elif p:
            fp += 1
        else:
            tn += 1
    return BinaryMetrics(tp=tp, fp=fp, tn=tn, fn=fn)


@dataclass(frozen=True)
class EvalMetrics:
    """``compute_metrics`` 결과. 모든 dict 의 언어 키에는 ``"all"`` 이 포함된다."""

    threshold: float
    n: int
    by_lang: dict[str, BinaryMetrics]
    false_positive: dict[str, dict[str, Rate]]  # lang -> {"benign", "hard_negative"} -> Rate
    evasion: dict[str, dict[str, Rate]]  # lang -> {기법..., "all"} -> 탐지율

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "n": self.n,
            "by_lang": {k: v.to_dict() for k, v in self.by_lang.items()},
            "false_positive": {
                lang: {k: asdict(r) for k, r in d.items()}
                for lang, d in self.false_positive.items()
            },
            "evasion": {
                lang: {k: asdict(r) for k, r in d.items()} for lang, d in self.evasion.items()
            },
        }


def compute_metrics(
    samples: Sequence[Sample], scores: Sequence[float], threshold: float
) -> EvalMetrics:
    """샘플과 탐지기 점수로 지표를 계산한다. ``score >= threshold`` 이면 공격 판정."""
    if len(samples) != len(scores):
        raise ValueError(f"샘플 {len(samples)}건과 점수 {len(scores)}건의 개수가 다릅니다")
    preds = [s >= threshold for s in scores]
    langs = [*sorted({s.lang for s in samples}), ALL]

    def in_lang(s: Sample, lang: str) -> bool:
        return lang == ALL or s.lang == lang

    by_lang: dict[str, BinaryMetrics] = {}
    false_positive: dict[str, dict[str, Rate]] = {}
    evasion: dict[str, dict[str, Rate]] = {}
    techniques = sorted({s.evasion for s in samples if s.evasion is not None})

    for lang in langs:
        rows = [(s, p) for s, p in zip(samples, preds, strict=True) if in_lang(s, lang)]
        originals = [(s, p) for s, p in rows if s.is_original]

        by_lang[lang] = binary_metrics([s.label for s, _ in originals], [p for _, p in originals])

        benign = [p for s, p in originals if s.label == 0 and not s.is_hard_negative]
        hard = [p for s, p in originals if s.is_hard_negative]
        false_positive[lang] = {
            "benign": rate(sum(benign), len(benign)),
            "hard_negative": rate(sum(hard), len(hard)),
        }

        variants = [(s, p) for s, p in rows if s.label == 1 and not s.is_original]
        per_tech = {}
        for tech in techniques:
            hits = [p for s, p in variants if s.evasion == tech]
            per_tech[tech] = rate(sum(hits), len(hits))
        per_tech[ALL] = rate(sum(p for _, p in variants), len(variants))
        evasion[lang] = per_tech

    return EvalMetrics(
        threshold=threshold,
        n=len(samples),
        by_lang=by_lang,
        false_positive=false_positive,
        evasion=evasion,
    )
