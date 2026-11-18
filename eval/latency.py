"""탐지기 지연시간 측정 (S5-3).

측정 방식:
    - **요청 1건 = ``predict([text])`` 1회** 로 잰다. 실서비스에서 Guard 는 요청마다 한 문장을
      검사하므로, 배치 처리량이 아니라 단건 지연시간이 기준이다.
    - 워밍업: 본 측정 전에 ``warmup`` 회 호출하고 버린다 (모델 지연 로딩·캐시·JIT 영향 제거).
    - 각 텍스트를 ``repeats`` 회 측정해 모두 표본으로 쓴다.
    - 전체와 **입력 길이(문자 수) 구간별** 로 평균 / p50 / p95 / 최대 (ms) 를 낸다.
    - 측정 환경(CPU·OS·Python)을 함께 기록한다. CPU 기준 측정이므로 GPU 를 쓰는
      어댑터는 CPU 로 설정해서 넘겨야 한다.

p95 등 백분위수는 선형 보간 (numpy ``percentile`` 기본값과 같은 방식).
"""

from __future__ import annotations

import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from eval.detector import Detector

__all__ = [
    "DEFAULT_LENGTH_BINS",
    "LatencyReport",
    "LatencyStats",
    "environment",
    "length_bin",
    "measure_latency",
    "percentile",
]

# (하한 이상, 상한 미만) 문자 수. test 세트 길이 분포: 중앙값 54자, 최대 약 1,000자.
DEFAULT_LENGTH_BINS: tuple[tuple[int, int | None], ...] = (
    (0, 32),
    (32, 128),
    (128, 512),
    (512, None),
)


def bin_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi - 1}"


def length_bin(n_chars: int, bins: Sequence[tuple[int, int | None]] = DEFAULT_LENGTH_BINS) -> str:
    """문자 수가 속한 구간 라벨 (예: ``"32-127"``, ``"512+"``)."""
    for lo, hi in bins:
        if n_chars >= lo and (hi is None or n_chars < hi):
            return bin_label(lo, hi)
    raise ValueError(f"길이 {n_chars} 가 어느 구간에도 속하지 않습니다: {bins}")


def percentile(values: Sequence[float], q: float) -> float:
    """선형 보간 백분위수 (``q`` 는 0~100)."""
    if not values:
        raise ValueError("빈 표본의 백분위수는 정의되지 않습니다")
    if not 0 <= q <= 100:
        raise ValueError(f"q 는 0~100 이어야 합니다: {q}")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class LatencyStats:
    """지연시간 요약 (단위 ms)."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> LatencyStats | None:
        if not samples_ms:
            return None
        return cls(
            n=len(samples_ms),
            mean_ms=sum(samples_ms) / len(samples_ms),
            p50_ms=percentile(samples_ms, 50),
            p95_ms=percentile(samples_ms, 95),
            max_ms=max(samples_ms),
        )


def environment() -> dict[str, str | int | None]:
    """측정 환경 기록용 정보."""
    return {
        "device": "cpu",
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class LatencyReport:
    """``measure_latency`` 결과. 표본이 없는 길이 구간은 ``None``."""

    detector: str
    warmup: int
    repeats: int
    overall: LatencyStats | None
    by_length: dict[str, LatencyStats | None]
    env: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "overall": asdict(self.overall) if self.overall else None,
            "by_length": {k: (asdict(v) if v else None) for k, v in self.by_length.items()},
            "env": dict(self.env),
        }


def measure_latency(
    detector: Detector,
    texts: Sequence[str],
    *,
    warmup: int = 20,
    repeats: int = 1,
    bins: Sequence[tuple[int, int | None]] = DEFAULT_LENGTH_BINS,
    timer: Callable[[], float] = time.perf_counter,
) -> LatencyReport:
    """``texts`` 각각을 단건 ``predict`` 로 측정한다.

    Args:
        warmup: 본 측정 전 버리는 호출 수. ``texts`` 를 앞에서부터 순환하며 사용한다.
        repeats: 텍스트당 측정 횟수 (모두 표본에 포함).
        bins: 길이 구간 ``(하한 이상, 상한 미만)`` 목록. 상한 ``None`` 은 무한대.
        timer: 초 단위 단조 시계 (테스트에서 주입).
    """
    if warmup < 0 or repeats < 1:
        raise ValueError(f"warmup >= 0, repeats >= 1 이어야 합니다: {warmup}, {repeats}")

    if texts:
        for i in range(warmup):
            detector.predict([texts[i % len(texts)]])

    labels = [bin_label(lo, hi) for lo, hi in bins]
    per_bin: dict[str, list[float]] = {label: [] for label in labels}
    all_ms: list[float] = []
    for text in texts:
        label = length_bin(len(text), bins)
        for _ in range(repeats):
            start = timer()
            detector.predict([text])
            ms = (timer() - start) * 1000.0
            all_ms.append(ms)
            per_bin[label].append(ms)

    return LatencyReport(
        detector=detector.name,
        warmup=warmup,
        repeats=repeats,
        overall=LatencyStats.from_samples(all_ms),
        by_length={label: LatencyStats.from_samples(per_bin[label]) for label in labels},
        env=environment(),
    )
