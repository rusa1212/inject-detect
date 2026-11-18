"""평가용 데이터 로더: ``data/processed/*.jsonl`` (S4-5 공통 스키마)을 읽는다.

``python -m data.build`` 로 생성한 파일을 읽기만 하므로
``data`` 패키지(pandas 등)에 의존하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

__all__ = ["HARD_NEGATIVE", "PROCESSED_DIR", "SPLITS", "Sample", "load_split"]

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLITS = ("train", "valid", "test")
HARD_NEGATIVE = "hard_negative"  # 하드 네거티브 행의 category 값


@dataclass(frozen=True)
class Sample:
    """평가 샘플 한 건 (S4-5 공통 스키마)."""

    id: str
    text: str
    label: int
    lang: str
    category: str
    source: str
    group: str
    evasion: str | None = None
    split: str | None = None

    @property
    def is_hard_negative(self) -> bool:
        return self.label == 0 and self.category == HARD_NEGATIVE

    @property
    def is_original(self) -> bool:
        """우회 변형이 아닌 원문인지."""
        return self.evasion is None


_FIELDS = {f.name for f in fields(Sample)}


def _parse(line: str, where: str) -> Sample:
    try:
        row = json.loads(line)
        sample = Sample(**{k: v for k, v in row.items() if k in _FIELDS})
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"{where}: 잘못된 행입니다 ({e})") from None
    if sample.label not in (0, 1):
        raise ValueError(f"{where}: label 은 0 또는 1 이어야 합니다: {sample.label!r}")
    return sample


def load_split(split: str = "test", data_dir: Path | str = PROCESSED_DIR) -> list[Sample]:
    """split 하나(``train``/``valid``/``test``) 또는 ``all`` (세 split 합본)을 읽는다."""
    names = SPLITS if split == "all" else (split,)
    if split != "all" and split not in SPLITS:
        raise ValueError(f"알 수 없는 split: {split!r} (가능: {', '.join(SPLITS)}, all)")
    data_dir = Path(data_dir)
    samples: list[Sample] = []
    for name in names:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 가 없습니다. 먼저 `python -m data.build` 로 데이터셋을 생성하세요."
            )
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    samples.append(_parse(line, f"{path.name}:{i}"))
    return samples
