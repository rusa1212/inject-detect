"""``python -m data.hard_negatives``: 검증 + 통계 + 현재 룰 엔진 오탐률 (참고용)."""

from __future__ import annotations

import sys
from collections import Counter

from data.hard_negatives import load_hard_negatives, stats


def main() -> int:
    rows = load_hard_negatives()
    print(f"[total] {len(rows)}")
    for section, counts in stats(rows).items():
        print(f"[{section}] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    from promptshield.normalize import normalize
    from promptshield.rules import RuleEngine

    engine = RuleEngine()
    fp = Counter(r.lang for r in rows if engine.scan(normalize(r.text)).flagged)
    total = Counter(r.lang for r in rows)
    for lang in sorted(total):
        print(f"[rule FP] {lang}: {fp[lang]}/{total[lang]} ({fp[lang] / total[lang]:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
