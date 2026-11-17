"""한국어 인젝션 YAML 로더·검증기.

사용법::

    python -m data.korean.dataset                  # 검증 + 통계 출력
    python -m data.korean.dataset --check-sources  # 번역 원문 ID가 deepset 원본에 있는지도 확인

YAML 형식은 ``data/korean/injections/*.yaml`` 상단 주석과 ``GUIDELINES.md`` 참고.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from promptshield.rules.schema import Category

INJECTIONS_DIR = Path(__file__).resolve().parent / "injections"
DEEPSET_RAW_DIR = Path(__file__).resolve().parents[1] / "raw" / "deepset_prompt_injections"

METHODS = frozenset({"translation", "variant", "original"})
STYLES = frozenset({"honorific", "plain", "colloquial"})
CATEGORIES = frozenset(c.value for c in Category)

_HANGUL = re.compile(r"[가-힣]")
_SOURCE_ID = re.compile(r"[0-9a-f]{12}")


class KoDatasetError(ValueError):
    """YAML 데이터 검증 실패."""


@dataclass(frozen=True)
class KoInjection:
    """한국어 인젝션 한 건 (S4-5 공통 스키마로 변환되기 전 형태)."""

    id: str  # <group>-<번호>, 예: t001-2
    text: str
    category: str
    method: str
    style: str
    group: str  # split 누수 방지 단위
    source_id: str | None  # 번역 원문 ID (직접 작성은 None)
    label: int = 1
    lang: str = "ko"


def source_id(text: str) -> str:
    """원문 텍스트 → 번역 추적용 ID (sha1 앞 12자리)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _norm(text: str) -> str:
    return " ".join(text.split())


def _parse_file(path: Path) -> list[KoInjection]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise KoDatasetError(f"{path.name}: version: 1 이 필요합니다")
    rows: list[KoInjection] = []
    for g in doc.get("groups") or []:
        gid = g.get("id")
        where = f"{path.name}:{gid}"
        if not gid or not isinstance(gid, str):
            raise KoDatasetError(f"{path.name}: 그룹 id 누락")
        cat = g.get("category")
        if cat not in CATEGORIES:
            raise KoDatasetError(f"{where}: 알 수 없는 category {cat!r}")
        src = g.get("source")
        if src is not None and not _SOURCE_ID.fullmatch(str(src)):
            raise KoDatasetError(f"{where}: source는 12자리 hex 또는 null")
        items = g.get("items") or []
        if not items:
            raise KoDatasetError(f"{where}: items 비어 있음")
        methods = set()
        for i, item in enumerate(items, 1):
            if not (isinstance(item, list) and len(item) == 3):
                raise KoDatasetError(f"{where}#{i}: [method, style, text] 형식이어야 함")
            method, style, text = item
            if method not in METHODS:
                raise KoDatasetError(f"{where}#{i}: 알 수 없는 method {method!r}")
            if style not in STYLES:
                raise KoDatasetError(f"{where}#{i}: 알 수 없는 style {style!r}")
            if not isinstance(text, str) or not text.strip():
                raise KoDatasetError(f"{where}#{i}: 빈 텍스트")
            if not _HANGUL.search(text):
                raise KoDatasetError(f"{where}#{i}: 한글이 없음")
            methods.add(method)
            rows.append(
                KoInjection(
                    id=f"{gid}-{i}",
                    text=text,
                    category=cat,
                    method=method,
                    style=style,
                    group=gid,
                    source_id=src,
                )
            )
        # 번역 그룹: source 필수 + translation 포함 / 직접 작성 그룹: source 없음 + original 포함
        if src is not None and ("translation" not in methods or "original" in methods):
            raise KoDatasetError(f"{where}: source가 있는 그룹은 translation(+variant)만 가능")
        if src is None and ("original" not in methods or "translation" in methods):
            raise KoDatasetError(f"{where}: source가 없는 그룹은 original(+variant)만 가능")
    return rows


def load_ko_injections(directory: Path = INJECTIONS_DIR) -> list[KoInjection]:
    """``directory/*.yaml``을 모두 읽어 검증된 행 목록을 돌려준다."""
    rows: list[KoInjection] = []
    for path in sorted(directory.glob("*.yaml")):
        rows.extend(_parse_file(path))
    groups = Counter(r.id.rsplit("-", 1)[0] for r in rows if r.id.endswith("-1"))
    dup_groups = [g for g, n in groups.items() if n > 1]
    if dup_groups:
        raise KoDatasetError(f"중복 그룹 id: {dup_groups}")
    seen: dict[str, str] = {}
    for r in rows:
        key = _norm(r.text)
        if key in seen:
            raise KoDatasetError(f"중복 텍스트: {r.id} == {seen[key]}")
        seen[key] = r.id
    return rows


def deepset_source_ids(raw_dir: Path = DEEPSET_RAW_DIR) -> set[str]:
    """내려받은 deepset 원본의 공격 문장 ID 집합 (``python -m data.download`` 선행 필요)."""
    import pandas as pd

    ids: set[str] = set()
    for path in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        ids.update(source_id(t) for t in df.loc[df["label"] == 1, "text"])
    if not ids:
        raise FileNotFoundError(f"deepset 원본이 없습니다: {raw_dir}")
    return ids


def stats(rows: list[KoInjection]) -> dict[str, dict[str, int]]:
    """카테고리·방법·문체별 건수."""
    return {
        "total": {"rows": len(rows), "groups": len({r.group for r in rows})},
        "category": dict(Counter(r.category for r in rows)),
        "method": dict(Counter(r.method for r in rows)),
        "style": dict(Counter(r.style for r in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="한국어 인젝션 데이터 검증·통계")
    parser.add_argument("--check-sources", action="store_true", help="번역 원문 ID 존재 확인")
    args = parser.parse_args(argv)

    rows = load_ko_injections()
    for section, counts in stats(rows).items():
        print(f"[{section}] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.check_sources:
        known = deepset_source_ids()
        missing = sorted({r.source_id for r in rows if r.source_id and r.source_id not in known})
        if missing:
            print(f"원본에 없는 source: {missing}", file=sys.stderr)
            return 1
        print("[sources] 모든 번역 원문 ID 확인됨")
    return 0


__all__ = ["KoDatasetError", "KoInjection", "load_ko_injections", "source_id", "stats"]

if __name__ == "__main__":
    sys.exit(main())
