"""학습 설정 (S7-1).

하이퍼파라미터는 YAML 파일(``training/configs/*.yaml``)로 관리하고, 실험 중 일부 값만 바꿀 때는
CLI ``--set key=value`` 로 덮어쓴다. 실제로 쓴 설정은 출력 폴더에 ``config.yaml`` 로 다시 저장된다.

    cfg = load_config("training/configs/mdeberta-v3-base.yaml", ["learning_rate=3e-5", "epochs=5"])
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

__all__ = ["METRIC_KEYS", "TrainConfig", "load_config", "parse_override"]

ROOT = Path(__file__).resolve().parent.parent

# best_metric 으로 쓸 수 있는 값: "<lang>.<지표>" (lang: ko | en | all)
METRIC_KEYS = ("precision", "recall", "f1")


@dataclass
class TrainConfig:
    # 모델·실행
    model_name: str = "microsoft/mdeberta-v3-base"
    run_name: str = "mdeberta-v3-base"
    output_dir: str = "training/outputs"  # 레포 미포함 (.gitignore)
    data_dir: str = "data/processed"
    seed: int = 42

    # 데이터
    max_length: int = 256  # train 기준 mDeBERTa p99 278, 98.7% 가 256 이하
    train_evasion: bool = True  # 우회 변형(base64·자모 등)을 학습에 포함
    langs: list[str] | None = None  # None = 전체 언어

    # 최적화
    epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_batch_size: int = 16
    eval_batch_size: int = 64
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    # None = 가중치 없음, "balanced" = 학습 데이터 라벨 빈도 역수, [w0, w1] = 직접 지정
    class_weight: str | list[float] | None = None
    fp16: bool = True  # CUDA 에서만 적용 (CPU 에서는 무시)

    # 평가·체크포인트
    threshold: float = 0.5
    best_metric: str = "ko.f1"  # valid 기준 최적 체크포인트 선택 지표
    save_model: bool = True

    # 스모크 테스트·디버깅용 (None = 제한 없음)
    max_steps: int | None = None
    limit_train: int | None = None
    limit_eval: int | None = None

    def __post_init__(self) -> None:
        # PyYAML(YAML 1.1)은 "3e-5" 처럼 소수점 없는 지수 표기를 문자열로 읽는다
        # → 숫자 필드를 선언된 형으로 변환
        for name in _FLOAT_FIELDS:
            setattr(self, name, float(getattr(self, name)))
        for name in _INT_FIELDS:
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, int(v))
        self.validate()

    def validate(self) -> None:
        lang, _, metric = self.best_metric.partition(".")
        if lang not in ("ko", "en", "all") or metric not in METRIC_KEYS:
            raise ValueError(
                f"best_metric 은 '<ko|en|all>.<{'|'.join(METRIC_KEYS)}>' 형식이어야 합니다: "
                f"{self.best_metric}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold 는 0~1 범위여야 합니다: {self.threshold}")
        if self.epochs < 1 or self.train_batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("epochs·train_batch_size·grad_accum_steps 는 1 이상이어야 합니다")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio 는 0 이상 1 미만이어야 합니다: {self.warmup_ratio}")
        cw = self.class_weight
        valid_list = isinstance(cw, list) and len(cw) == 2 and all(float(w) > 0 for w in cw)
        if cw is not None and cw != "balanced" and not valid_list:
            raise ValueError(f"class_weight 는 null, 'balanced', [w0, w1] 중 하나여야 합니다: {cw}")

    def resolve_path(self, p: str) -> Path:
        """상대 경로는 레포 루트 기준으로 해석한다 (Colab 등 작업 폴더와 무관하게)."""
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    @property
    def run_dir(self) -> Path:
        return self.resolve_path(self.output_dir) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


_FLOAT_FIELDS = ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm", "threshold")
_INT_FIELDS = (
    "seed",
    "max_length",
    "epochs",
    "train_batch_size",
    "eval_batch_size",
    "grad_accum_steps",
    "max_steps",
    "limit_train",
    "limit_eval",
)


def parse_override(item: str) -> tuple[str, object]:
    """``key=value`` → (key, YAML 로 해석한 값). 예) ``class_weight=[1,2]``, ``langs=null``."""
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"--set 은 key=value 형식이어야 합니다: {item!r}")
    return key.strip(), yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> TrainConfig:
    """YAML 설정 + ``key=value`` 덮어쓰기 → ``TrainConfig``. 모르는 키는 오류."""
    data: dict = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"설정 파일 최상위는 매핑이어야 합니다: {path}")
        data.update(loaded or {})
    for item in overrides:
        k, v = parse_override(item)
        data[k] = v
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"알 수 없는 설정 키: {unknown}")
    return TrainConfig(**data)
