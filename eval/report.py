"""평가 실행·결과 저장·Markdown 결과표 생성 (S5-4).

결과 JSON 구조 (``eval/results/<detector>_<split>.json``)::

    {
      "detector": "rules", "split": "test", "threshold": 0.5,
      "created_at": "2026-11-18T12:00:00+00:00",
      "dataset": {"n": 508, "langs": {"en": 119, "ko": 389}},
      "metrics": {...},   # EvalMetrics.to_dict()  (S5-2)
      "latency": {...}    # LatencyReport.to_dict() (S5-3), 측정 생략 시 null
    }

Markdown 은 이 JSON 만으로 다시 만들 수 있다 (``render_markdown(json.load(...))``).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from eval.dataset import Sample
from eval.detector import Detector, check_scores
from eval.latency import measure_latency
from eval.metrics import ALL, compute_metrics

__all__ = ["RESULTS_DIR", "render_markdown", "run_eval", "save_result"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_eval(
    detector: Detector,
    samples: Sequence[Sample],
    *,
    split: str,
    threshold: float | None = None,
    latency: bool = True,
    warmup: int = 20,
    repeats: int = 1,
) -> dict:
    """탐지기로 샘플 전체를 채점해 지표·지연시간 결과 dict 를 만든다.

    Args:
        threshold: 판정 임계값. 생략하면 ``detector.threshold``.
        latency: False 면 지연시간 측정을 건너뛴다 (``"latency": None``).
    """
    texts = [s.text for s in samples]
    scores = check_scores(detector.predict(texts), len(texts), name=detector.name)
    thr = detector.threshold if threshold is None else float(threshold)
    metrics = compute_metrics(samples, scores, thr)
    lat = measure_latency(detector, texts, warmup=warmup, repeats=repeats) if latency else None
    return {
        "detector": detector.name,
        "split": split,
        "threshold": thr,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "n": len(samples),
            "langs": dict(sorted(Counter(s.lang for s in samples).items())),
        },
        "metrics": metrics.to_dict(),
        "latency": lat.to_dict() if lat else None,
    }


def save_result(result: dict, out_dir: Path | str = RESULTS_DIR) -> tuple[Path, Path]:
    """결과를 ``<detector>_<split>.json`` / ``.md`` 로 저장하고 두 경로를 돌려준다 (덮어씀)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['detector']}_{result['split']}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return json_path, md_path


# ---------------------------------------------------------------- Markdown


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _rate(r: dict) -> str:
    """``12.5% (1/8) [2.2–47.1]`` — 비율, 건수, Wilson 95% CI (%)."""
    if r["n"] == 0:
        return "- (0/0)"
    return (
        f"{_pct(r['value'])} ({r['k']}/{r['n']}) [{r['ci_low'] * 100:.1f}–{r['ci_high'] * 100:.1f}]"
    )


def _ms(x: float) -> str:
    return f"{x:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def _lang_label(lang: str) -> str:
    return "전체" if lang == ALL else lang.upper()


def render_markdown(result: dict) -> str:
    """결과 dict 를 Markdown 결과표로 만든다."""
    m = result["metrics"]
    ds = result["dataset"]
    langs_n = ", ".join(f"{k.upper()} {v}" for k, v in ds["langs"].items())
    lines = [
        f"# 평가 결과: `{result['detector']}` ({result['split']})",
        "",
        f"- 임계값: {result['threshold']} (score ≥ 임계값 → 공격)",
        f"- 데이터: {ds['n']}건 ({langs_n})",
        f"- 생성: {result['created_at']}",
        "",
        "## 언어별 성능 (원문만, 우회 변형 제외)",
        "",
    ]
    lines += _table(
        ["언어", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"],
        [
            [
                _lang_label(lang),
                f"{b['precision']:.3f}",
                f"{b['recall']:.3f}",
                f"{b['f1']:.3f}",
                *(str(b[k]) for k in ("tp", "fp", "tn", "fn")),
            ]
            for lang, b in m["by_lang"].items()
        ],
    )

    lines += ["", "## 오탐률", "", "비율 (오탐/전체) [Wilson 95% CI, %]", ""]
    lines += _table(
        ["언어", "정상", "하드 네거티브"],
        [
            [_lang_label(lang), _rate(d["benign"]), _rate(d["hard_negative"])]
            for lang, d in m["false_positive"].items()
        ],
    )

    techniques = [t for t in next(iter(m["evasion"].values()), {}) if t != ALL]
    lines += ["", "## 우회 공격 탐지율", "", "비율 (탐지/전체) [Wilson 95% CI, %]", ""]
    if techniques:
        lines += _table(
            ["언어", *techniques, "전체"],
            [
                [_lang_label(lang), *(_rate(d[t]) for t in techniques), _rate(d[ALL])]
                for lang, d in m["evasion"].items()
            ],
        )
    else:
        lines.append("우회 변형 샘플 없음")

    lines += ["", "## 지연시간 (CPU, 단건 predict, ms)", ""]
    lat = result.get("latency")
    if not lat or not lat.get("overall"):
        lines.append("측정 생략")
    else:
        rows = [("전체", lat["overall"])] + [(f"{k}자", v) for k, v in lat["by_length"].items()]
        lines += _table(
            ["입력 길이", "n", "평균", "p50", "p95", "최대"],
            [
                [label, "0", "-", "-", "-", "-"]
                if st is None
                else [
                    label,
                    str(st["n"]),
                    _ms(st["mean_ms"]),
                    _ms(st["p50_ms"]),
                    _ms(st["p95_ms"]),
                    _ms(st["max_ms"]),
                ]
                for label, st in rows
            ],
        )
        env = lat.get("env", {})
        lines += [
            "",
            f"워밍업 {lat['warmup']}회 제외, 텍스트당 {lat['repeats']}회 측정. "
            f"환경: {env.get('processor', '?')} ({env.get('cpu_count', '?')} threads), "
            f"{env.get('platform', '?')}, Python {env.get('python', '?')}",
        ]
    return "\n".join(lines) + "\n"
