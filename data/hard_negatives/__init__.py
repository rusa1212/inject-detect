"""하드 네거티브 세트 (S4-4): 인젝션 키워드가 들어 있지만 정상(label=0)인 문장.

사용법::

    python -m data.hard_negatives    # 검증 + 통계 + 현재 룰 엔진 오탐률

YAML 형식은 ``ko.yaml`` 상단 주석 참고. 직접 작성한 데이터라 레포에 포함된다.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

HARD_NEGATIVES_DIR = Path(__file__).resolve().parent

TYPES = frozenset(
    {
        "security_edu",
        "injection_question",
        "keyword_everyday",
        "benign_roleplay",
        "prompt_engineering",
        "it_support",
    }
)
STYLES = frozenset({"honorific", "plain", "colloquial"})
LANGS = frozenset({"ko", "en"})

_HANGUL = re.compile(r"[가-힣]")


class HardNegativeError(ValueError):
    """하드 네거티브 YAML 검증 실패."""


@dataclass(frozen=True)
class HardNegative:
    id: str  # hn-<lang>-<번호>
    text: str
    lang: str
    type: str
    style: str
    label: int = 0

    @property
    def group(self) -> str:
        """split 단위. 하드 네거티브는 문장마다 독립이다."""
        return self.id


def _parse_file(path: Path) -> list[HardNegative]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise HardNegativeError(f"{path.name}: version: 1 이 필요합니다")
    lang = doc.get("lang")
    if lang not in LANGS:
        raise HardNegativeError(f"{path.name}: lang은 {sorted(LANGS)} 중 하나")
    rows: list[HardNegative] = []
    for i, item in enumerate(doc.get("items") or [], 1):
        where = f"{path.name}#{i}"
        if not (isinstance(item, list) and len(item) == 3):
            raise HardNegativeError(f"{where}: [type, style, text] 형식이어야 함")
        typ, style, text = item
        if typ not in TYPES:
            raise HardNegativeError(f"{where}: 알 수 없는 type {typ!r}")
        if style not in STYLES:
            raise HardNegativeError(f"{where}: 알 수 없는 style {style!r}")
        if not isinstance(text, str) or not text.strip():
            raise HardNegativeError(f"{where}: 빈 텍스트")
        if (lang == "ko") != bool(_HANGUL.search(text)):
            raise HardNegativeError(f"{where}: lang={lang}와 텍스트 언어가 맞지 않음")
        rows.append(
            HardNegative(id=f"hn-{lang}-{i:03d}", text=text, lang=lang, type=typ, style=style)
        )
    return rows


def load_hard_negatives(directory: Path = HARD_NEGATIVES_DIR) -> list[HardNegative]:
    """``directory/*.yaml``을 읽어 검증된 행 목록을 돌려준다."""
    rows: list[HardNegative] = []
    for path in sorted(directory.glob("*.yaml")):
        rows.extend(_parse_file(path))
    seen: dict[str, str] = {}
    for r in rows:
        key = " ".join(r.text.split())
        if key in seen:
            raise HardNegativeError(f"중복 텍스트: {r.id} == {seen[key]}")
        seen[key] = r.id
    return rows


def stats(rows: list[HardNegative]) -> dict[str, dict[str, int]]:
    return {
        "lang": dict(Counter(r.lang for r in rows)),
        "type": dict(Counter(r.type for r in rows)),
        "style": dict(Counter(r.style for r in rows)),
    }


__all__ = ["HardNegative", "HardNegativeError", "load_hard_negatives", "stats"]
