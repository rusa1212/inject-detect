"""입력 정규화: 우회 기법을 표준 형태로 되돌려 룰·모델이 같은 텍스트를 보게 한다.

S2-1 범위(유니코드 정규화):
    1. 보이지 않는 문자·제어문자 제거 (유니코드 태그 문자는 ASCII 로 복원)
    2. NFKC 정규화 (전각→반각, 수학 기호 문자→일반 문자 등). 단, 한글 호환 자모는 보호
    3. 동형문자(homoglyph) 매핑 (키릴/그리스 등 → 라틴)
    4. 대소문자 통일(casefold) 및 연속 공백 정리

인코딩 디코딩(S2-2), 한국어 특화 정규화(S2-3), ``normalize()`` 공개 인터페이스(S2-4)는
이후 세그먼트에서 추가한다.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "HOMOGLYPHS",
    "collapse_whitespace",
    "fold_case",
    "map_homoglyphs",
    "nfkc",
    "normalize_unicode",
    "strip_invisible",
]

# ---------------------------------------------------------------------------
# 1. 보이지 않는 문자 / 제어문자
# ---------------------------------------------------------------------------

# 일반 범주(Cf: format)에 속하지 않지만 화면에 보이지 않아 우회에 쓰이는 문자
_EXTRA_INVISIBLE = frozenset(
    {
        "ᅟ",  # HANGUL CHOSEONG FILLER
        "ᅠ",  # HANGUL JUNGSEONG FILLER
        "ㅤ",  # HANGUL FILLER
        "ﾠ",  # HALFWIDTH HANGUL FILLER
        "͏",  # COMBINING GRAPHEME JOINER (Mn)
        "⠀",  # BRAILLE PATTERN BLANK
        "᠋",  # MONGOLIAN FREE VARIATION SELECTOR ONE (Mn)
        "᠌",
        "᠍",
        "᠏",
    }
)
# 변이 선택자(VS1–VS16, VS17–VS256): 글자 뒤에 붙어도 보이지 않음
_VARIATION_SELECTORS = re.compile("[︀-️\U000e0100-\U000e01ef]")

# 공백으로 취급할 제어문자 (줄바꿈/탭은 지우지 않고 공백으로 바꿔 단어가 붙지 않게 함)
_WHITESPACE_CONTROLS = frozenset("\t\n\v\f\r\x1c\x1d\x1e\x1f\x85")

# 유니코드 태그 문자(U+E0020–U+E007E)는 보이지 않지만 ASCII 와 1:1 대응되어
# "ASCII smuggling" 으로 숨은 지시문을 넣는 데 쓰인다 → 지우지 않고 ASCII 로 복원
_TAG_BASE = 0xE0000
_TAG_PRINTABLE = range(0xE0020, 0xE007F)


def strip_invisible(text: str) -> str:
    """제로폭 문자·서식 문자(Cf)·제어문자(Cc)를 제거한다.

    - 탭/줄바꿈 등 공백성 제어문자는 공백 한 칸으로 바꾼다.
    - 유니코드 태그 문자는 대응하는 ASCII 문자로 복원한다.
    """
    text = _VARIATION_SELECTORS.sub("", text)
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _TAG_PRINTABLE:
            out.append(chr(cp - _TAG_BASE))
        elif ch in _WHITESPACE_CONTROLS:
            out.append(" ")
        elif ch in _EXTRA_INVISIBLE:
            continue
        elif unicodedata.category(ch) in ("Cf", "Cc"):
            # Cf: U+200B–U+200F, U+202A–U+202E, U+2060–U+2064, U+FEFF, U+00AD, 나머지 태그 문자 등
            continue
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# 2. NFKC (한글 호환 자모 보호)
# ---------------------------------------------------------------------------

# NFKC 는 호환 자모(ㄱ U+3131 등)를 조합형 자모(U+1100 대)로 바꾸고 다시 음절로 합성해서
# "ㅇㅣㅈㅓㄴ" → "이저ᄂ" 처럼 받침을 잘못 처리한 깨진 문자열을 만든다.
# 자모 결합은 S2-3 에서 올바르게 하므로, 여기서는 호환 자모를 그대로 둔다.
_COMPAT_JAMO = re.compile("([ㄱ-ㆎ]+)")


def _halfwidth_jamo_to_compat(ch: str) -> str:
    """반각 한글 자모(U+FFA1–U+FFDC)를 같은 이름의 호환 자모로 바꾼다."""
    name = unicodedata.name(ch, "")
    if name.startswith("HALFWIDTH HANGUL LETTER "):
        try:
            return unicodedata.lookup(name.removeprefix("HALFWIDTH "))
        except KeyError:
            return ch
    return ch


_HALFWIDTH_JAMO = re.compile("[ﾡ-ￜ]")


def nfkc(text: str) -> str:
    """NFKC 정규화. 전각→반각, 합자·원문자·수학 문자 등을 기본 문자로 바꾼다.

    한글 호환 자모는 NFKC 대상에서 제외하고, 반각 자모는 호환 자모로 바꾼다.
    """
    text = _HALFWIDTH_JAMO.sub(lambda m: _halfwidth_jamo_to_compat(m.group()), text)
    parts = _COMPAT_JAMO.split(text)
    # split 결과에서 홀수 인덱스가 캡처된 자모 구간
    return "".join(
        part if i % 2 else unicodedata.normalize("NFKC", part) for i, part in enumerate(parts)
    )


# ---------------------------------------------------------------------------
# 3. 동형문자(homoglyph) 매핑
# ---------------------------------------------------------------------------

# 라틴 문자와 모양이 같거나 거의 같은 문자 → 라틴 문자.
# NFKC 이후에 적용하므로 전각·수학 문자 등 NFKC 가 처리하는 것은 넣지 않는다.
HOMOGLYPHS: dict[str, str] = {
    # --- 키릴 소문자 ---
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h",
    "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s",
    "і": "i", "ї": "i", "ј": "j", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w",
    "ɡ": "g", "ӏ": "l", "ь": "b", "ѵ": "v", "ү": "y", "ғ": "f",
    # --- 키릴 대문자 ---
    "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S",
    "І": "I", "Ї": "I", "Ј": "J", "Ԁ": "D", "Һ": "H", "Ԛ": "Q", "Ԝ": "W",
    "Ӏ": "I", "Ү": "Y",
    # --- 그리스 소문자 ---
    "α": "a", "β": "b", "γ": "y", "ε": "e", "η": "n", "ι": "i", "κ": "k",
    "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w",
    "ϲ": "c", "ϳ": "j",
    # --- 그리스 대문자 ---
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "Ϲ": "C",
    # --- 기타 라틴 확장/IPA ---
    "ı": "i", "ȷ": "j", "ɑ": "a", "ɩ": "i", "ɪ": "i", "ʟ": "l", "ɴ": "n",
    "ʀ": "r", "ʏ": "y", "ᴀ": "a", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ᴊ": "j",
    "ᴋ": "k", "ᴍ": "m", "ᴏ": "o", "ᴘ": "p", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v",
    "ᴡ": "w", "ᴢ": "z", "ꜱ": "s",
    # --- 아르메니아 ---
    "օ": "o", "ս": "u", "ց": "g", "հ": "h", "ո": "n",
}  # fmt: skip

_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)


def map_homoglyphs(text: str) -> str:
    """동형문자를 라틴 문자로 바꾼다 (예: 키릴 ``іgnоrе`` → ``ignore``)."""
    return text.translate(_HOMOGLYPH_TABLE)


# ---------------------------------------------------------------------------
# 4. 대소문자 · 공백
# ---------------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")


def fold_case(text: str) -> str:
    """대소문자 구분을 없앤다 (``str.casefold``: ``ß`` → ``ss`` 등 포함)."""
    return text.casefold()


def collapse_whitespace(text: str) -> str:
    """연속 공백(줄바꿈 포함)을 공백 한 칸으로 줄이고 양끝 공백을 제거한다."""
    return _WS_RUN.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """S2-1 유니코드 정규화 파이프라인.

    순서: 보이지 않는 문자 제거 → NFKC → 동형문자 매핑 → casefold → 공백 정리
    """
    text = strip_invisible(text)
    text = nfkc(text)
    text = map_homoglyphs(text)
    text = fold_case(text)
    return collapse_whitespace(text)
