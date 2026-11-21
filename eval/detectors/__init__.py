"""기본 탐지기 어댑터. import 시 각 어댑터가 레지스트리에 등록된다."""

from eval.detectors.hf import HFClassifierDetector
from eval.detectors.rules import RuleDetector

__all__ = ["HFClassifierDetector", "RuleDetector"]
