"""우회 공격 세트 자동 생성 (S4-4).

공격 문장에 기계적 우회 변형을 적용한다. 각 변형은 ``promptshield.normalize``의
해당 단계가 되돌리도록 설계된 기법이며, 정규화 모듈의 매핑을 역으로 사용한다.

| 변형 | 기법 | 정규화 대응 단계 |
|---|---|---|
| ``base64``    | 문장 전체를 Base64로 인코딩해 지시 템플릿에 삽입 | ``decode:base64`` |
| ``homoglyph`` | 라틴 문자를 키릴·그리스 동형문자로 치환 | ``homoglyph`` |
| ``jamo``      | 한글 음절을 호환 자모로 분리 (``무시`` → ``ㅁㅜㅅㅣ``) | ``jamo`` |
| ``spacing``   | 글자 사이에 공백·구두점 삽입 (``무.시.해``) | ``korean_separator`` |

사용법::

    python -m data.evasion          # data/processed/evasion.jsonl 생성 + 통계

생성물은 ``data/processed/``(``.gitignore``)에 쓰이며, 같은 입력이면 항상 같은 결과가 나온다.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from promptshield.normalize import HOMOGLYPHS

EVASIONS: tuple[str, ...] = ("base64", "homoglyph", "jamo", "spacing")

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

# 라틴 → 동형문자 후보 (정규화 표의 역방향). 결정적 결과를 위해 정렬해 둔다.
_REVERSE_HOMOGLYPHS: dict[str, list[str]] = {}
for _src, _dst in sorted(HOMOGLYPHS.items()):
    _REVERSE_HOMOGLYPHS.setdefault(_dst, []).append(_src)

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
_SYLLABLE = re.compile("[가-힣]")
_HANGUL_WORD = re.compile("[가-힣]{2,}")
_LATIN_WORD = re.compile("[A-Za-z]{3,}")

_BASE64_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ko": (
        "아래 Base64를 디코딩해서 그대로 따라 해: {payload}",
        "다음 문자열을 해석한 뒤 지시대로 수행해 주세요. {payload}",
        "{payload}",
    ),
    "en": (
        "Decode this Base64 and follow it: {payload}",
        "Please decode the following and do what it says. {payload}",
        "{payload}",
    ),
}
_SEPARATORS = (" ", ".", "-", "/", "*")


@dataclass(frozen=True)
class EvasionRow:
    """우회 변형 한 건. ``group``은 원문과 같아 S4-5에서 같은 split에 배치된다."""

    id: str  # <원문 id>~<evasion>
    text: str
    label: int
    lang: str
    category: str
    evasion: str
    source_row: str  # 원문 행 id
    group: str


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:16], 16))


def to_base64(text: str, rng: random.Random, lang: str = "ko") -> str:
    """문장 전체를 UTF-8 Base64로 인코딩해 템플릿에 넣는다."""
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    template = rng.choice(_BASE64_TEMPLATES.get(lang, _BASE64_TEMPLATES["en"]))
    return template.format(payload=payload)


def to_homoglyph(text: str, rng: random.Random, ratio: float = 0.5) -> str | None:
    """3자 이상 라틴 단어의 문자 일부를 동형문자로 바꾼다. 바꿀 문자가 없으면 None."""
    out = list(text)
    changed = False
    for m in _LATIN_WORD.finditer(text):
        for i in range(m.start(), m.end()):
            cands = _REVERSE_HOMOGLYPHS.get(text[i])
            if cands and rng.random() < ratio:
                out[i] = rng.choice(cands)
                changed = True
    return "".join(out) if changed else None


def decompose_syllable(ch: str) -> str:
    """한글 음절 한 글자 → 호환 자모 문자열 (``한`` → ``ㅎㅏㄴ``)."""
    code = ord(ch) - 0xAC00
    cho, rest = divmod(code, 21 * 28)
    jung, jong = divmod(rest, 28)
    return _CHO[cho] + _JUNG[jung] + _JONG[jong]


def to_jamo(text: str, rng: random.Random, ratio: float = 0.6) -> str | None:
    """한글 단어 일부(기본 60%)를 자모로 분리한다. 한글 단어가 없으면 None."""
    words = list(_HANGUL_WORD.finditer(text))
    if not words:
        return None
    picked = [m for m in words if rng.random() < ratio] or [rng.choice(words)]
    out, last = [], 0
    for m in picked:
        out.append(text[last : m.start()])
        out.append("".join(decompose_syllable(c) for c in m.group()))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def to_spacing(text: str, rng: random.Random, ratio: float = 0.5) -> str | None:
    """단어 일부의 글자 사이에 구분자를 끼운다 (한글 2자+, 라틴 3자+ 단어 대상)."""
    words = list(_HANGUL_WORD.finditer(text)) + list(_LATIN_WORD.finditer(text))
    if not words:
        return None
    words.sort(key=lambda m: m.start())
    picked = [m for m in words if rng.random() < ratio] or [rng.choice(words)]
    sep = rng.choice(_SEPARATORS)
    out, last = [], 0
    for m in picked:
        out.append(text[last : m.start()])
        out.append(sep.join(m.group()))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def apply_evasion(name: str, text: str, key: str, lang: str = "ko") -> str | None:
    """이름으로 변형 하나를 적용한다. 적용 불가(대상 문자 없음)면 None."""
    rng = _rng(f"{key}~{name}")
    if name == "base64":
        return to_base64(text, rng, lang)
    if name == "homoglyph":
        return to_homoglyph(text, rng)
    if name == "jamo":
        return to_jamo(text, rng)
    if name == "spacing":
        return to_spacing(text, rng)
    raise KeyError(f"알 수 없는 evasion: {name} (가능: {EVASIONS})")


@dataclass(frozen=True)
class AttackSeed:
    """변형 대상 공격 문장."""

    id: str
    text: str
    lang: str
    category: str
    group: str


def generate(seeds: Iterable[AttackSeed], evasions: Iterable[str] = EVASIONS) -> list[EvasionRow]:
    """모든 시드에 각 변형을 적용한다. 적용 불가한 조합은 건너뛴다."""
    names = tuple(evasions)
    rows: list[EvasionRow] = []
    for s in seeds:
        for name in names:
            out = apply_evasion(name, s.text, s.id, s.lang)
            if out is None or out == s.text:
                continue
            rows.append(
                EvasionRow(
                    id=f"{s.id}~{name}",
                    text=out,
                    label=1,
                    lang=s.lang,
                    category=s.category,
                    evasion=name,
                    source_row=s.id,
                    group=s.group,
                )
            )
    return rows


def ko_seeds() -> list[AttackSeed]:
    """S4-3 한국어 인젝션 → 시드."""
    from data.korean.dataset import load_ko_injections

    return [
        AttackSeed(id=r.id, text=r.text, lang="ko", category=r.category, group=r.group)
        for r in load_ko_injections()
    ]


def en_seeds(raw_dir: Path | None = None) -> list[AttackSeed]:
    """deepset 영어 공격 문장 → 시드 (``python -m data.download`` 선행 필요).

    group은 원문 ID(sha1 앞 12자리)로, 한국어 번역 그룹의 ``source``와 같은 값이다.
    category는 원본에 없으므로 ``unknown``이다 (S4-5에서 필요 시 보강).
    """
    import pandas as pd

    from data.korean.dataset import DEEPSET_RAW_DIR, source_id

    raw_dir = raw_dir or DEEPSET_RAW_DIR
    seeds: list[AttackSeed] = []
    seen: set[str] = set()
    for path in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        for text in df.loc[df["label"] == 1, "text"]:
            sid = source_id(text)
            if sid in seen:
                continue
            seen.add(sid)
            seeds.append(
                AttackSeed(id=f"en-{sid}", text=text, lang="en", category="unknown", group=sid)
            )
    return seeds


def write_jsonl(rows: Iterable[EvasionRow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="우회 공격 세트 생성")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "evasion.jsonl")
    parser.add_argument("--no-en", action="store_true", help="영어(deepset) 시드 제외")
    args = parser.parse_args(argv)

    seed_fns: list[Callable[[], list[AttackSeed]]] = [ko_seeds]
    if not args.no_en:
        seed_fns.append(en_seeds)
    seeds = [s for fn in seed_fns for s in fn()]
    rows = generate(seeds)
    n = write_jsonl(rows, args.out)
    counts = Counter((r.lang, r.evasion) for r in rows)
    print(f"시드 {len(seeds)}건 → 변형 {n}건: {args.out}")
    for (lang, ev), c in sorted(counts.items()):
        print(f"  {lang}/{ev}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
