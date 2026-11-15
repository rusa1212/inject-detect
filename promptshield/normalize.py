"""입력 정규화: 우회 기법을 표준 형태로 되돌려 룰·모델이 같은 텍스트를 보게 한다.

S2-1 범위(유니코드 정규화):
    1. 보이지 않는 문자·제어문자 제거 (유니코드 태그 문자는 ASCII 로 복원)
    2. NFKC 정규화 (전각→반각, 수학 기호 문자→일반 문자 등). 단, 한글 호환 자모는 보호
    3. 동형문자(homoglyph) 매핑 (키릴/그리스 등 → 라틴)
    4. 대소문자 통일(casefold) 및 연속 공백 정리

S2-2 범위(인코딩 디코딩):
    Base64 / hex / URL 인코딩 구간을 찾아 디코딩하고, 디코딩 결과를 원문 뒤에 덧붙인다.
    디코딩 깊이·스캔 길이·구간 수·출력 길이를 ``DecodeLimits`` 로 제한한다 (DoS 방지).
    Base64 는 대소문자를 구분하므로 ``fold_case`` 보다 **먼저** 실행해야 한다.

한국어 특화 정규화(S2-3), ``normalize()`` 공개 인터페이스(S2-4)는
이후 세그먼트에서 추가한다.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote_plus

__all__ = [
    "HOMOGLYPHS",
    "DecodeLimits",
    "DecodedSegment",
    "append_decoded",
    "collapse_whitespace",
    "decode_layers",
    "find_encoded",
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


# ---------------------------------------------------------------------------
# 5. 인코딩 디코딩 (S2-2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeLimits:
    """디코딩 자원 제한 (DoS 방지).

    Attributes:
        max_depth: 중첩 디코딩 최대 깊이 (예: base64 안의 base64). 1 이면 한 겹만.
        max_scan_chars: 한 텍스트에서 인코딩 구간을 찾을 최대 길이. 넘는 부분은 스캔하지 않는다.
        max_segment_chars: 인코딩 구간 하나에서 디코딩할 최대 길이. 넘으면 앞부분만 디코딩한다.
        max_segments: 전체 깊이를 통틀어 디코딩할 최대 구간 수.
        max_total_decoded_chars: 디코딩 결과 문자 수 합계 상한.
        min_base64_chars: Base64 후보로 볼 최소 길이 (짧은 영단어 오탐 방지).
        min_hex_chars: 연속 hex 문자열 후보로 볼 최소 길이 (짝수 자리).
    """

    max_depth: int = 3
    max_scan_chars: int = 20_000
    max_segment_chars: int = 8_192
    max_segments: int = 32
    max_total_decoded_chars: int = 32_768
    min_base64_chars: int = 16
    min_hex_chars: int = 16


DEFAULT_DECODE_LIMITS = DecodeLimits()


@dataclass(frozen=True)
class DecodedSegment:
    """디코딩된 구간 하나.

    ``start``/``end`` 는 이 구간을 찾은 텍스트 기준 위치다. ``depth == 1`` 이면 원문,
    ``depth >= 2`` 이면 바로 윗단계 디코딩 결과(``DecodedSegment.decoded``) 기준이다.
    """

    encoding: str  # "base64" | "hex" | "url"
    start: int
    end: int
    depth: int
    decoded: str


# 앞뒤가 같은 문자 집합이 아닐 때만 구간으로 본다 (단어 중간 매칭 방지)
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{2,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_CONTIG_RE = re.compile(r"(?<![0-9A-Za-z])(?:0x)?((?:[0-9A-Fa-f]{2})+)(?![0-9A-Za-z])")
_HEX_ESCAPED_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){4,}")
_HEX_SEPARATED_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[0-9A-Fa-f]{2}[ :,-]){3,}[0-9A-Fa-f]{2}(?![0-9A-Za-z])"
)
# 공백 단위 토큰. `\S*%XX\S*` 형태는 긴 무공백 입력에서 O(n^2) 백트래킹이 생기므로
# 토큰만 잘라내고 % 이스케이프 개수는 _decode_url 에서 확인한다.
_URL_TOKEN_RE = re.compile(r"\S+")
_URL_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_HEX_DIGITS_RE = re.compile(r"[0-9A-Fa-f]{2}")


def _looks_like_text(s: str) -> bool:
    """디코딩 결과가 사람이 읽을 수 있는 텍스트인지 (바이너리·쓰레기 결과 배제)."""
    if len(s.strip()) < 2:
        return False
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\r\t")
    if printable / len(s) < 0.95:
        return False
    # 글자(한글 포함)·숫자가 최소 절반은 되어야 함
    alnum = sum(1 for ch in s if ch.isalnum() or ch.isspace())
    return alnum / len(s) >= 0.6


def _bytes_to_text(data: bytes) -> str | None:
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return s if _looks_like_text(s) else None


def _decode_base64(token: str, limits: DecodeLimits) -> str | None:
    body = token.rstrip("=")
    if len(body) < limits.min_base64_chars:
        return None
    # 순수 소문자/순수 대문자/순수 숫자 등은 일반 단어·숫자일 가능성이 높음 → 최소 2종류 섞여야 함
    kinds = sum(
        (
            any(c.islower() for c in body),
            any(c.isupper() for c in body),
            any(c.isdigit() or c in "+/-_" for c in body),
        )
    )
    if kinds < 2:
        return None
    body = body[: limits.max_segment_chars]
    body = body[: len(body) - len(body) % 4] if len(body) % 4 == 1 else body
    body = body.replace("-", "+").replace("_", "/")
    try:
        data = base64.b64decode(body + "=" * (-len(body) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return _bytes_to_text(data)


def _decode_hex(digits: str, limits: DecodeLimits) -> str | None:
    digits = digits[: limits.max_segment_chars]
    digits = digits[: len(digits) - len(digits) % 2]
    try:
        data = bytes.fromhex(digits)
    except ValueError:
        return None
    return _bytes_to_text(data)


def _decode_url(token: str, limits: DecodeLimits) -> str | None:
    if len(_URL_ESCAPE_RE.findall(token)) < 2:
        return None
    s = unquote_plus(token[: limits.max_segment_chars], errors="strict")
    if s == token or not _looks_like_text(s):
        return None
    return s


def find_encoded(
    text: str, limits: DecodeLimits = DEFAULT_DECODE_LIMITS, *, depth: int = 1
) -> list[DecodedSegment]:
    """``text`` 한 겹에서 인코딩 구간을 찾아 디코딩한다 (재귀 없음).

    hex → URL → Base64 순서로 찾고, 앞에서 채택된 구간과 겹치는 후보는 버린다.
    결과는 시작 위치 순으로 정렬된다.
    """
    scan = text[: limits.max_scan_chars]
    found: list[DecodedSegment] = []
    taken: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e in taken)

    def add(encoding: str, a: int, b: int, decoded: str | None) -> None:
        if decoded is None or overlaps(a, b):
            return
        taken.append((a, b))
        found.append(DecodedSegment(encoding, a, b, depth, decoded))

    for m in _HEX_ESCAPED_RE.finditer(scan):
        add("hex", m.start(), m.end(), _decode_hex(m.group().replace("\\x", ""), limits))
    for m in _HEX_SEPARATED_RE.finditer(scan):
        digits = "".join(_HEX_DIGITS_RE.findall(m.group()))
        add("hex", m.start(), m.end(), _decode_hex(digits, limits))
    for m in _HEX_CONTIG_RE.finditer(scan):
        if len(m.group(1)) >= limits.min_hex_chars:
            add("hex", m.start(), m.end(), _decode_hex(m.group(1), limits))
    for m in _URL_TOKEN_RE.finditer(scan):
        add("url", m.start(), m.end(), _decode_url(m.group(), limits))
    for m in _B64_RE.finditer(scan):
        add("base64", m.start(), m.end(), _decode_base64(m.group(), limits))

    found.sort(key=lambda seg: seg.start)
    return found


def decode_layers(text: str, limits: DecodeLimits = DEFAULT_DECODE_LIMITS) -> list[DecodedSegment]:
    """인코딩 구간을 ``limits.max_depth`` 겹까지 재귀적으로 디코딩한다.

    너비 우선으로 진행하며, 구간 수(``max_segments``)나 출력 길이
    (``max_total_decoded_chars``) 한도에 닿으면 즉시 멈춘다.
    """
    results: list[DecodedSegment] = []
    total_chars = 0
    frontier = [text]
    for depth in range(1, limits.max_depth + 1):
        next_frontier: list[str] = []
        for source in frontier:
            for seg in find_encoded(source, limits, depth=depth):
                if len(results) >= limits.max_segments:
                    return results
                remaining = limits.max_total_decoded_chars - total_chars
                if remaining <= 0:
                    return results
                if len(seg.decoded) > remaining:
                    seg = DecodedSegment(
                        seg.encoding, seg.start, seg.end, seg.depth, seg.decoded[:remaining]
                    )
                results.append(seg)
                total_chars += len(seg.decoded)
                next_frontier.append(seg.decoded)
        if not next_frontier:
            break
        frontier = next_frontier
    return results


def append_decoded(text: str, limits: DecodeLimits = DEFAULT_DECODE_LIMITS) -> str:
    """원문 뒤에 디코딩 결과를 줄바꿈으로 이어 붙인다 (인코딩 구간이 없으면 원문 그대로).

    원문을 지우지 않고 덧붙이므로, 원문 자체의 패턴과 숨겨진 페이로드를 모두 검사할 수 있다.
    """
    segments = decode_layers(text, limits)
    if not segments:
        return text
    return "\n".join([text, *(seg.decoded for seg in segments)])
