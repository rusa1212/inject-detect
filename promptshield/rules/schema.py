"""룰 스키마: YAML 룰 한 건을 표현하는 자료형.

YAML 룰 파일 포맷 (``version: 1``)::

    version: 1
    rules:
      - id: en-override-001          # 필수, 전체 룰셋에서 유일
        lang: en                      # 필수, en | ko | any
        category: instruction_override  # 필수, Category 값 중 하나
        pattern: 'ignore\\s+(all\\s+)?previous\\s+instructions'  # 필수, 정규식
        weight: 0.9                   # 필수, 0 < weight <= 1
        description: 이전 지시 무시    # 선택
        enabled: true                 # 선택, 기본 true

패턴은 ``normalize()`` 를 거친 텍스트(casefold·공백 정리 완료)에 매칭된다고 가정하고
소문자로 작성한다. 컴파일 시 ``re.IGNORECASE`` 를 함께 걸어 두지만 이는 안전장치일 뿐이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["LANGS", "SCHEMA_VERSION", "Category", "Rule"]

SCHEMA_VERSION = 1

LANGS: frozenset[str] = frozenset({"en", "ko", "any"})


class Category(str, Enum):
    """공격 카테고리."""

    INSTRUCTION_OVERRIDE = "instruction_override"  # 지시 무시
    ROLE_HIJACK = "role_hijack"  # 역할 탈취
    JAILBREAK = "jailbreak"  # 탈옥
    PROMPT_LEAK = "prompt_leak"  # 시스템 프롬프트 유출 요구


@dataclass(frozen=True)
class Rule:
    """사전 컴파일된 정규식을 가진 룰 한 건."""

    id: str
    lang: str
    category: Category
    pattern: str
    weight: float
    description: str = ""
    enabled: bool = True
    regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # frozen 이므로 object.__setattr__ 로 컴파일 결과를 채운다.
        # 잘못된 정규식이면 re.error 가 그대로 올라간다 (로더가 RuleLoadError 로 감싼다).
        object.__setattr__(self, "regex", re.compile(self.pattern, re.IGNORECASE))

    def search(self, text: str) -> re.Match[str] | None:
        """텍스트에서 첫 매칭을 찾는다."""
        return self.regex.search(text)
