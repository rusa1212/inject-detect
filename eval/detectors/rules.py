"""룰 엔진(S3) Detector 어댑터.

``normalize=False`` 로 만들면 정규화 없이 원문에 룰을 적용한다
(S11-1 정규화 on/off 비교용).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from eval.detector import register_detector
from promptshield.normalize import NormalizedText
from promptshield.rules import RuleEngine

__all__ = ["RuleDetector"]


class RuleDetector:
    """``RuleEngine.scan(text).score`` 를 점수로 쓰는 탐지기."""

    def __init__(
        self,
        engine: RuleEngine | None = None,
        *,
        langs: Iterable[str] | None = None,
        threshold: float | None = None,
        normalize: bool = True,
    ) -> None:
        if engine is not None and langs is not None:
            raise ValueError("engine 과 langs 는 함께 지정할 수 없습니다")
        self.engine = engine if engine is not None else RuleEngine(langs=langs)
        self.threshold = self.engine.threshold if threshold is None else float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold 는 0~1 범위여야 합니다: {self.threshold}")
        self.normalize = normalize
        self.name = "rules" if normalize else "rules-raw"

    def predict(self, texts: Sequence[str]) -> list[float]:
        if self.normalize:
            return [self.engine.scan(t).score for t in texts]
        return [self.engine.scan(NormalizedText(original=t, text=t)).score for t in texts]


@register_detector("rules")
def _rules(**kwargs) -> RuleDetector:
    return RuleDetector(**kwargs)


@register_detector("rules-raw")
def _rules_raw(**kwargs) -> RuleDetector:
    return RuleDetector(normalize=False, **kwargs)
