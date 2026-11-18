"""평가 CLI (S5-4).

    python -m eval.run --detector rules                 # test split 평가 → 결과표 출력 + 저장
    python -m eval.run --detector rules-raw --split valid
    python -m eval.run --detector rules --no-latency --no-save
    python -m eval.run --list                           # 등록된 탐지기 목록

결과는 ``eval/results/<detector>_<split>.json`` / ``.md`` 에 저장된다 (``--out-dir`` 로 변경).
"""

from __future__ import annotations

import argparse
import sys

from eval.dataset import PROCESSED_DIR, SPLITS, load_split
from eval.detector import available_detectors, get_detector
from eval.report import RESULTS_DIR, render_markdown, run_eval, save_result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.run", description="탐지기 평가")
    p.add_argument("--detector", help="탐지기 이름 (--list 로 확인)")
    p.add_argument("--list", action="store_true", help="등록된 탐지기 목록 출력")
    p.add_argument("--split", default="test", choices=[*SPLITS, "all"], help="기본 test")
    p.add_argument("--threshold", type=float, help="판정 임계값 (기본: 탐지기 기본값)")
    p.add_argument("--data-dir", default=str(PROCESSED_DIR), help="data/processed 경로")
    p.add_argument("--out-dir", default=str(RESULTS_DIR), help="결과 저장 경로")
    p.add_argument("--no-save", action="store_true", help="결과 파일을 저장하지 않음")
    p.add_argument("--no-latency", action="store_true", help="지연시간 측정 생략")
    p.add_argument("--warmup", type=int, default=20, help="지연시간 워밍업 호출 수")
    p.add_argument("--repeats", type=int, default=1, help="지연시간 텍스트당 측정 횟수")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows 콘솔/파이프에서도 한글·기호가 깨지지 않도록
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(available_detectors()))
        return 0
    if not args.detector:
        parser.error("--detector 가 필요합니다 (--list 로 목록 확인)")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold 는 0~1 범위여야 합니다: {args.threshold}")

    try:
        detector = get_detector(args.detector)
        samples = load_split(args.split, args.data_dir)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"오류: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2

    result = run_eval(
        detector,
        samples,
        split=args.split,
        threshold=args.threshold,
        latency=not args.no_latency,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(render_markdown(result), end="")
    if not args.no_save:
        json_path, md_path = save_result(result, args.out_dir)
        print(f"\n저장: {json_path}\n      {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
