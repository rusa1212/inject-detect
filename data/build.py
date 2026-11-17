"""전체 데이터셋 빌드 (S4-5).

사용법::

    python -m data.build                 # 원본 다운로드(없을 때만) → 통합 → 분할 → 저장
    python -m data.build --no-evasion    # 우회 변형 제외

산출물 (``data/processed/``, 레포 미포함)::

    train.jsonl / valid.jsonl / test.jsonl   공통 스키마 행
    stats.json                               분할·언어·라벨·카테고리별 통계

공통 스키마: ``text``, ``label``(1=공격, 0=정상), ``lang``(ko|en), ``category``, ``source``,
``split`` + 추적용 ``id``, ``group``, ``evasion``.

파이프라인:
    1. 수집: deepset(영어만), S4-3 한국어 인젝션, S4-4 하드 네거티브, dolly·klue 정상 후보
    2. 중복 제거: 정규화 완전 일치 삭제 + MinHash 근접 중복(Jaccard ≥ 0.85, 같은 라벨) 삭제
       + 포함 관계(containment ≥ 0.9) 쌍은 같은 group으로 병합
    3. 정상 샘플링: 언어별 공격 1건당 정상 ``BENIGN_PER_ATTACK``건 (하드 네거티브는 별도 추가)
    4. 우회 변형: 남은 공격 문장에 S4-4 생성기 적용 (원문 group 상속)
    5. 분할: group 해시로 train/valid/test = 80/10/10.
       원문-번역-변형은 group을 공유하므로 같은 split에 들어간다
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from data.sources import BENIGN_PER_ATTACK

# numpy·pandas는 빌드할 때만 필요하므로 함수 안에서 import한다 (``pip install -e ".[data]"``).
# 모듈 import만으로는 런타임 최소 의존성(pyyaml) 외에 아무것도 요구하지 않는다.

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SPLITS: tuple[tuple[str, float], ...] = (("train", 0.8), ("valid", 0.9), ("test", 1.0))
NEAR_DUP_JACCARD = 0.85
CONTAINMENT = 0.9

# 중복 시 남길 우선순위 (작을수록 우선): 직접 구축 데이터 > 공개 공격 데이터 > 정상 말뭉치
SOURCE_PRIORITY = {
    "ko_injection": 0,
    "hard_negative": 1,
    "deepset": 2,
    "dolly_15k": 3,
    "klue_mrc": 4,
}


@dataclass(frozen=True)
class Row:
    """공통 스키마 행."""

    id: str
    text: str
    label: int
    lang: str
    category: str
    source: str
    group: str
    evasion: str | None = None
    split: str | None = None


# ---------------------------------------------------------------------------
# 1. 수집
# ---------------------------------------------------------------------------

_DE_WORDS = frozenset([
    "und", "der", "die", "das", "ist", "nicht", "ich", "sie", "wie", "was", "für", "mit", "auf",
    "ein", "eine", "einen", "zu", "den", "dem", "von", "es", "sind", "wir", "du", "bitte", "warum",
    "wer", "kann", "mir", "dich", "dir", "sich", "auch", "noch", "nach", "über", "aus", "bei",
    "wird", "werden", "habe", "hat", "gibt", "welche", "vergiss", "alles", "alle", "jetzt", "nun",
    "schreibe",
])  # fmt: skip
_EN_WORDS = frozenset([
    "the", "and", "is", "you", "to", "of", "what", "how", "a", "in", "for", "are", "do", "i", "it",
    "that", "this", "with", "can", "my", "your", "be", "on", "me", "which", "have", "why", "who",
    "should", "does", "about", "did", "when",
])  # fmt: skip
_WORD = re.compile(r"[a-zäöüß]+")
_UMLAUT = re.compile(r"[äöüß]")


def deepset_lang(text: str) -> str:
    """deepset 문장 언어 판정: ``en`` | ``de`` | ``mixed`` (불용어·움라우트 휴리스틱).

    deepset은 영어·독일어가 대부분이라, 독일어 근거가 뚜렷하지 않으면 영어로 본다.
    """
    lower = text.lower()
    words = _WORD.findall(lower)
    de = sum(w in _DE_WORDS for w in words) + 2 * len(_UMLAUT.findall(lower))
    en = sum(w in _EN_WORDS for w in words)
    if de >= 2 and en >= 2 and max(de, en) < 2 * min(de, en):
        return "mixed"
    if de >= 2 and de > en:
        return "de"
    return "en"


def _sha_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def collect_deepset(raw_dir: Path = RAW_DIR) -> list[Row]:
    import pandas as pd

    from data.korean.dataset import source_id

    rows: list[Row] = []
    for path in sorted((raw_dir / "deepset_prompt_injections").glob("*.parquet")):
        for text, label in pd.read_parquet(path)[["text", "label"]].itertuples(index=False):
            if deepset_lang(text) != "en":
                continue
            sid = source_id(text)
            rows.append(
                Row(
                    id=f"deepset-{sid}",
                    text=text,
                    label=int(label),
                    lang="en",
                    category="unknown" if label == 1 else "benign",
                    source="deepset",
                    group=sid,  # 한국어 번역 그룹의 source와 같은 값 → 같은 split
                )
            )
    return rows


def collect_ko_injections() -> list[Row]:
    from data.korean.dataset import load_ko_injections

    return [
        Row(
            id=r.id,
            text=r.text,
            label=1,
            lang="ko",
            category=r.category,
            source="ko_injection",
            group=r.source_id or r.group,  # 번역 그룹은 영어 원문과 같은 group
        )
        for r in load_ko_injections()
    ]


def collect_hard_negatives() -> list[Row]:
    from data.hard_negatives import load_hard_negatives

    return [
        Row(
            id=r.id,
            text=r.text,
            label=0,
            lang=r.lang,
            category="hard_negative",
            source="hard_negative",
            group=r.group,
        )
        for r in load_hard_negatives()
    ]


def _take_stable(rows: list[Row], n: int) -> list[Row]:
    """텍스트 해시 순으로 정렬해 앞에서 n개 (입력 순서와 무관한 결정적 샘플)."""
    return sorted(rows, key=lambda r: _sha_key(r.text))[: max(n, 0)]


def collect_dolly(raw_dir: Path = RAW_DIR) -> list[Row]:
    import pandas as pd

    df = pd.read_json(raw_dir / "dolly_15k" / "train.jsonl", lines=True)
    return [
        Row(
            id=f"dolly-{i}",
            text=t.strip(),
            label=0,
            lang="en",
            category="benign",
            source="dolly_15k",
            group=f"dolly-{i}",
        )
        for i, t in enumerate(df["instruction"])
        if isinstance(t, str) and t.strip()
    ]


def collect_klue(raw_dir: Path = RAW_DIR) -> list[Row]:
    import pandas as pd

    rows: list[Row] = []
    for path in sorted((raw_dir / "klue_mrc").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["guid", "question"])
        for guid, q in df.itertuples(index=False):
            rows.append(
                Row(
                    id=f"klue-{guid}",
                    text=q.strip(),
                    label=0,
                    lang="ko",
                    category="benign",
                    source="klue_mrc",
                    group=f"klue-{guid}",
                )
            )
    return rows


# ---------------------------------------------------------------------------
# 2. 중복 제거
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def dedup_key(text: str) -> str:
    """완전 중복 판정 키: casefold + 구두점·공백 제거."""
    return _PUNCT.sub("", text.casefold())


def shingles(text: str, k: int = 5) -> frozenset[str]:
    s = " ".join(_PUNCT.sub(" ", text.casefold()).split())
    if len(s) <= k:
        return frozenset({s})
    return frozenset(s[i : i + k] for i in range(len(s) - k + 1))


_NUM_PERM = 64
_BANDS = 16
_PRIME = 4294967311  # 2^32보다 큰 소수
_MINHASH_SEED = 20261117


@lru_cache(maxsize=1)
def _minhash_params():
    """MinHash 해시 계수 (a, b). numpy를 처음 쓸 때 한 번만 만든다."""
    import numpy as np

    rng = np.random.default_rng(_MINHASH_SEED)
    a = rng.integers(1, 2**31, size=_NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, 2**31, size=_NUM_PERM, dtype=np.uint64)
    return a, b


def _minhash_signatures(shingle_sets: list[frozenset[str]]):
    """shingle 집합 목록 → (n, _NUM_PERM) uint64 MinHash 서명 행렬."""
    import numpy as np

    a, b = _minhash_params()
    prime = np.uint64(_PRIME)
    sigs = np.empty((len(shingle_sets), _NUM_PERM), dtype=np.uint64)
    for i, sh in enumerate(shingle_sets):
        x = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))
        sigs[i] = ((np.outer(a, x) + b[:, None]) % prime).min(axis=1)
    return sigs


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / min(len(a), len(b))


@dataclass
class DedupReport:
    exact: int = 0
    near: int = 0
    merged_groups: int = 0
    label_conflicts: int = 0


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        return True


def deduplicate(rows: Iterable[Row]) -> tuple[list[Row], DedupReport]:
    """완전 중복·근접 중복 제거 + 포함 관계 group 병합.

    - 우선순위(``SOURCE_PRIORITY``)가 높은 행을 남긴다.
    - 라벨이 다른 완전 중복은 모호하므로 양쪽 모두 삭제한다 (``label_conflicts``).
    - 근접 중복은 같은 라벨일 때만 삭제한다. 라벨이 다르면 하드 네거티브 대비 쌍일 수 있어 남긴다.
    - 포함 관계(짧은 문장이 긴 문장에 거의 통째로 들어 있음) 쌍은 삭제 대신 group을 합쳐
      같은 split에 들어가게 한다 (deepset의 '정상 질문 + 공격' 이어붙이기 대응).
    """
    report = DedupReport()
    ordered = sorted(rows, key=lambda r: (SOURCE_PRIORITY.get(r.source, 9), r.id))

    # (1) 완전 중복
    by_key: dict[str, list[Row]] = defaultdict(list)
    for r in ordered:
        by_key[dedup_key(r.text)].append(r)
    kept: list[Row] = []
    for dups in by_key.values():
        if len({r.label for r in dups}) > 1:
            report.label_conflicts += len(dups)
            continue
        kept.append(dups[0])
        report.exact += len(dups) - 1
    kept.sort(key=lambda r: (SOURCE_PRIORITY.get(r.source, 9), r.id))

    # (2) MinHash LSH 후보 → 정확한 Jaccard·containment 확인
    sh = [shingles(r.text) for r in kept]
    sigs = _minhash_signatures(sh)
    rows_per_band = _NUM_PERM // _BANDS
    candidates: set[tuple[int, int]] = set()
    for b in range(_BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        band = sigs[:, b * rows_per_band : (b + 1) * rows_per_band]
        for i, sig in enumerate(band):
            buckets[sig.tobytes()].append(i)
        for idx in buckets.values():
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    candidates.add((idx[x], idx[y]))

    drop: set[int] = set()
    uf = _UnionFind()
    for i, j in sorted(candidates):  # i < j → i가 우선순위 높음
        if i in drop or j in drop:
            continue
        a, b = kept[i], kept[j]
        if a.label == b.label and _jaccard(sh[i], sh[j]) >= NEAR_DUP_JACCARD:
            drop.add(j)
            report.near += 1
        elif a.group != b.group and _containment(sh[i], sh[j]) >= CONTAINMENT:
            report.merged_groups += uf.union(a.group, b.group)

    out = [replace(r, group=uf.find(r.group)) for k, r in enumerate(kept) if k not in drop]
    return out, report


# ---------------------------------------------------------------------------
# 3~5. 샘플링 · 우회 변형 · 분할
# ---------------------------------------------------------------------------


def benign_targets(rows: Iterable[Row]) -> dict[str, int]:
    """언어별 일반 정상 문장 목표 수 = 공격 수 × 비율 (하드 네거티브는 제외하고 계산)."""
    attacks = Counter(r.lang for r in rows if r.label == 1 and r.evasion is None)
    return {lang: n * BENIGN_PER_ATTACK for lang, n in attacks.items()}


def surface_key(text: str) -> str:
    """우회 변형용 중복 키: casefold + 공백 정리만 한다.

    ``dedup_key``는 구두점·공백을 지우므로 ``무/시/해``(spacing 변형)를 원문과 같다고 본다.
    변형은 바로 그 표면 차이가 핵심이라 별도 키를 쓴다.
    """
    return " ".join(text.casefold().split())


def evasion_rows(rows: Iterable[Row], existing_keys: set[str]) -> list[Row]:
    """공격 행에 우회 변형을 적용한다. 기존 행과 표면이 완전히 같은 변형은 버린다.

    ``existing_keys``는 ``surface_key`` 기준 집합이다.
    """
    from data.evasion import AttackSeed, generate

    base = {r.id: r for r in rows if r.label == 1 and r.evasion is None}
    seeds = [
        AttackSeed(id=r.id, text=r.text, lang=r.lang, category=r.category, group=r.group)
        for r in base.values()
    ]
    out: list[Row] = []
    for e in generate(seeds):
        key = surface_key(e.text)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        src = base[e.source_row]
        out.append(
            Row(
                id=e.id,
                text=e.text,
                label=1,
                lang=e.lang,
                category=e.category,
                source=src.source,
                group=e.group,
                evasion=e.evasion,
            )
        )
    return out


def split_of(group: str) -> str:
    """group 해시 → split (결정적). 같은 group은 항상 같은 split."""
    u = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    return next(name for name, upper in SPLITS if u < upper)


def assign_splits(rows: Iterable[Row]) -> list[Row]:
    return [replace(r, split=split_of(r.group)) for r in rows]


def build(include_evasion: bool = True, raw_dir: Path = RAW_DIR) -> tuple[list[Row], dict]:
    """전체 파이프라인을 실행해 (행 목록, 통계)를 돌려준다. 파일은 쓰지 않는다."""
    core = collect_ko_injections() + collect_hard_negatives() + collect_deepset(raw_dir)
    # 정상 후보는 목표의 1.3배를 미리 뽑아 두고, 중복 제거 후 목표 수만큼 자른다
    pre_targets = benign_targets(core)
    deepset_benign_en = sum(1 for r in core if r.source == "deepset" and r.label == 0)
    pools = {
        "en": _take_stable(collect_dolly(raw_dir), int(pre_targets.get("en", 0) * 1.3)),
        "ko": _take_stable(collect_klue(raw_dir), int(pre_targets.get("ko", 0) * 1.3)),
    }
    deduped, report = deduplicate(core + pools["en"] + pools["ko"])

    targets = benign_targets(deduped)
    targets["en"] = max(targets.get("en", 0) - deepset_benign_en, 0)  # deepset 정상도 일반 정상
    curated = [r for r in deduped if r.source not in ("dolly_15k", "klue_mrc")]
    benign = []
    for lang, src in (("en", "dolly_15k"), ("ko", "klue_mrc")):
        pool = [r for r in deduped if r.source == src]
        benign += _take_stable(pool, targets.get(lang, 0))
    rows = curated + benign

    if include_evasion:
        rows += evasion_rows(rows, {surface_key(r.text) for r in rows})

    rows = sorted(assign_splits(rows), key=lambda r: (r.split, r.lang, r.id))
    return rows, make_stats(rows, report)


def make_stats(rows: list[Row], report: DedupReport | None = None) -> dict:
    def count(key):
        return dict(sorted(Counter(key(r) for r in rows).items()))

    stats = {
        "total": len(rows),
        "split": count(lambda r: r.split),
        "split_lang_label": count(lambda r: f"{r.split}/{r.lang}/{r.label}"),
        "lang_label": count(lambda r: f"{r.lang}/{r.label}"),
        "category": count(lambda r: f"{r.lang}/{r.category}"),
        "source": count(lambda r: r.source),
        "evasion": count(lambda r: r.evasion or "none"),
        "groups": len({r.group for r in rows}),
    }
    if report is not None:
        stats["dedup"] = asdict(report)
    return stats


def write_outputs(rows: list[Row], stats: dict, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name, _ in SPLITS}
    try:
        for r in rows:
            files[r.split].write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    finally:
        for f in files.values():
            f.close()
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ensure_raw(raw_dir: Path = RAW_DIR) -> None:
    """원본이 없을 때만 내려받는다."""
    from data.download import download_all
    from data.sources import ALL_SOURCES

    missing = [
        s.name
        for s in ALL_SOURCES
        if not all(
            (raw_dir / s.name / f"{split}{Path(p).suffix}").exists() for split, p in s.files.items()
        )
    ]
    if missing:
        print(f"원본 다운로드: {missing}")
        download_all(missing, raw_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="전체 데이터셋 빌드")
    parser.add_argument("--no-evasion", action="store_true", help="우회 변형 제외")
    parser.add_argument("--no-download", action="store_true", help="원본 다운로드 생략")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)

    if not args.no_download:
        ensure_raw()
    rows, stats = build(include_evasion=not args.no_evasion)
    write_outputs(rows, stats, args.out)
    print(f"총 {stats['total']}건 → {args.out}")
    for key in ("split", "lang_label", "evasion", "dedup"):
        print(f"  {key}: {stats.get(key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
