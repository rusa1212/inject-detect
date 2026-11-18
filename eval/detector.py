"""평가 프레임워크의 공통 Detector 인터페이스 (S5-1).

룰·공개 모델·자체 모델·하이브리드를 같은 방식으로 평가하기 위한 최소 계약::

    class Detector(Protocol):
        name: str           # 결과표·파일명에 쓰는 식별자 (예: "rules")
        threshold: float    # score >= threshold 이면 공격(1)으로 판정
        def predict(self, texts: Sequence[str]) -> list[float]: ...

- ``predict`` 는 입력과 같은 길이·순서의 **0~1 점수**를 돌려준다 (높을수록 공격).
  점수를 받는 이유: 임계값 스윕·PR 곡선·하이브리드 결합(S8)을 탐지기 밖에서 할 수 있도록.
- 배치(``Sequence[str]``)로 받는다. 모델 어댑터는 배치 추론을, 룰 어댑터는 반복 호출을 한다.
- 정규화는 탐지기 책임이다. 평가 프레임워크는 원문을 넘긴다.

탐지기는 이름으로 등록해 두고 ``get_detector("rules")`` 로 만든다.
무거운 의존성(onnxruntime 등)은 팩토리 안에서 import 해 등록만으로는 로딩되지 않게 한다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

__all__ = [
    "Detector",
    "DetectorFactory",
    "available_detectors",
    "check_scores",
    "get_detector",
    "register_detector",
]


@runtime_checkable
class Detector(Protocol):
    """모든 탐지기가 따르는 프로토콜."""

    name: str
    threshold: float

    def predict(self, texts: Sequence[str]) -> list[float]:
        """각 텍스트의 공격 점수(0~1)를 입력 순서대로 돌려준다."""
        ...


DetectorFactory = Callable[..., Detector]

_REGISTRY: dict[str, DetectorFactory] = {}


def register_detector(name: str) -> Callable[[DetectorFactory], DetectorFactory]:
    """``name`` 으로 탐지기 팩토리를 등록하는 데코레이터."""

    def deco(factory: DetectorFactory) -> DetectorFactory:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"이미 등록된 탐지기 이름입니다: {name}")
        _REGISTRY[name] = factory
        return factory

    return deco


def _ensure_builtin() -> None:
    # 기본 어댑터 모듈을 import 하면 각 모듈의 @register_detector 가 실행된다.
    import eval.detectors  # noqa: F401


def available_detectors() -> list[str]:
    """등록된 탐지기 이름 (정렬)."""
    _ensure_builtin()
    return sorted(_REGISTRY)


def get_detector(name: str, **kwargs) -> Detector:
    """이름으로 탐지기를 만든다. ``kwargs`` 는 팩토리에 그대로 전달된다."""
    _ensure_builtin()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"알 수 없는 탐지기: {name!r} (사용 가능: {', '.join(sorted(_REGISTRY))})"
        ) from None
    detector = factory(**kwargs)
    if not isinstance(detector, Detector):
        raise TypeError(f"{name!r} 팩토리가 Detector 프로토콜을 만족하지 않는 객체를 반환했습니다")
    return detector


def check_scores(scores: Sequence[float], n: int, *, name: str = "detector") -> list[float]:
    """``predict`` 결과를 검증한다: 길이 ``n``, 유한한 0~1 실수.

    통과하면 float 리스트로 돌려준다.
    """
    out = [float(s) for s in scores]
    if len(out) != n:
        raise ValueError(f"{name}: 점수 개수 {len(out)} != 입력 개수 {n}")
    for i, s in enumerate(out):
        if not math.isfinite(s) or not 0.0 <= s <= 1.0:
            raise ValueError(f"{name}: {i}번째 점수가 0~1 범위를 벗어났습니다: {s}")
    return out
