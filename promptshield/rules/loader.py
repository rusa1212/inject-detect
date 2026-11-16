"""룰 로더: YAML 룰 파일을 읽어 검증하고 정규식을 사전 컴파일한다.

- ``load_rules(path)``: 파일 하나 또는 디렉터리(그 안의 ``*.yaml``/``*.yml``, 이름순)를 읽는다.
- ``load_default_rules()``: 패키지에 포함된 기본 룰셋(``promptshield/rules/data``)을 읽는다.
- 필수 필드 누락, 알 수 없는 필드, 잘못된 lang/category, weight 범위 위반,
  정규식 컴파일 실패, 전체 룰셋 내 id 중복은 모두 ``RuleLoadError`` 로 보고한다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from promptshield.rules.schema import LANGS, SCHEMA_VERSION, Category, Rule

__all__ = [
    "DEFAULT_RULES_DIR",
    "RuleLoadError",
    "load_default_rules",
    "load_rules",
    "parse_rules",
]

DEFAULT_RULES_DIR = Path(__file__).parent / "data"

_REQUIRED = ("id", "lang", "category", "pattern", "weight")
_OPTIONAL = ("description", "enabled")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class RuleLoadError(ValueError):
    """룰 파일이 스키마를 위반했을 때."""


def _fail(source: str, idx: int | None, msg: str) -> RuleLoadError:
    where = source if idx is None else f"{source} rules[{idx}]"
    return RuleLoadError(f"{where}: {msg}")


def _parse_rule(raw: Any, source: str, idx: int) -> Rule:
    if not isinstance(raw, dict):
        raise _fail(source, idx, "룰은 매핑이어야 합니다")

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise _fail(source, idx, f"필수 필드 누락: {', '.join(missing)}")
    unknown = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise _fail(source, idx, f"알 수 없는 필드: {', '.join(unknown)}")

    rule_id = raw["id"]
    if not isinstance(rule_id, str) or not _ID_RE.match(rule_id):
        raise _fail(source, idx, f"id 는 소문자·숫자·'-'·'_' 문자열이어야 합니다: {rule_id!r}")

    lang = raw["lang"]
    if lang not in LANGS:
        raise _fail(source, idx, f"[{rule_id}] lang 은 {sorted(LANGS)} 중 하나: {lang!r}")

    try:
        category = Category(raw["category"])
    except ValueError:
        allowed = [c.value for c in Category]
        raise _fail(
            source, idx, f"[{rule_id}] category 는 {allowed} 중 하나: {raw['category']!r}"
        ) from None

    pattern = raw["pattern"]
    if not isinstance(pattern, str) or not pattern.strip():
        raise _fail(source, idx, f"[{rule_id}] pattern 은 비어 있지 않은 문자열이어야 합니다")

    weight = raw["weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight <= 1:
        raise _fail(source, idx, f"[{rule_id}] weight 는 0 초과 1 이하 숫자: {weight!r}")

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise _fail(source, idx, f"[{rule_id}] description 은 문자열이어야 합니다")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise _fail(source, idx, f"[{rule_id}] enabled 는 true/false 여야 합니다")

    try:
        return Rule(
            id=rule_id,
            lang=lang,
            category=category,
            pattern=pattern,
            weight=float(weight),
            description=description,
            enabled=enabled,
        )
    except re.error as e:
        raise _fail(source, idx, f"[{rule_id}] 정규식 컴파일 실패: {e}") from e


def parse_rules(doc: Any, source: str = "<string>") -> list[Rule]:
    """``yaml.safe_load`` 결과(문서 하나)를 룰 목록으로 변환한다. id 중복은 검사하지 않는다."""
    if not isinstance(doc, dict):
        raise _fail(source, None, "최상위는 'version' 과 'rules' 를 가진 매핑이어야 합니다")
    if doc.get("version") != SCHEMA_VERSION:
        raise _fail(source, None, f"지원하지 않는 version: {doc.get('version')!r}")
    rules = doc.get("rules")
    if not isinstance(rules, list):
        raise _fail(source, None, "'rules' 는 리스트여야 합니다")
    return [_parse_rule(raw, source, i) for i, raw in enumerate(rules)]


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(p for p in path.iterdir() if p.suffix in (".yaml", ".yml"))
    elif path.is_file():
        yield path
    else:
        raise RuleLoadError(f"룰 경로를 찾을 수 없습니다: {path}")


def load_rules(path: str | Path, *, include_disabled: bool = False) -> list[Rule]:
    """파일 또는 디렉터리에서 룰을 읽는다. 기본적으로 ``enabled: false`` 룰은 제외한다."""
    rules: list[Rule] = []
    seen: dict[str, str] = {}
    for file in _iter_files(Path(path)):
        try:
            doc = yaml.safe_load(file.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise RuleLoadError(f"{file}: YAML 파싱 실패: {e}") from e
        for rule in parse_rules(doc, str(file)):
            if rule.id in seen:
                raise RuleLoadError(
                    f"{file}: 중복된 룰 id {rule.id!r} (처음 정의: {seen[rule.id]})"
                )
            seen[rule.id] = str(file)
            if rule.enabled or include_disabled:
                rules.append(rule)
    return rules


def load_default_rules(*, include_disabled: bool = False) -> list[Rule]:
    """패키지 기본 룰셋을 읽는다."""
    return load_rules(DEFAULT_RULES_DIR, include_disabled=include_disabled)
