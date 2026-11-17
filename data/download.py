"""HuggingFace Hub 데이터셋 원본 다운로드.

사용법::

    python -m data.download            # 전체 소스
    python -m data.download deepset_prompt_injections

원본은 ``data/raw/<source>/<split><원본 확장자>`` (예: ``train.parquet``, ``train.jsonl``)에
저장되며 레포에 커밋하지 않는다 (``.gitignore``).
``huggingface_hub`` 필요: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

from data.sources import ALL_SOURCES, HFSource

RAW_DIR = Path(__file__).resolve().parent / "raw"


def download_source(source: HFSource, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """소스의 모든 split 파일을 고정 revision으로 내려받아 ``raw_dir/<name>/``에 저장.

    Returns:
        split 이름 → 저장된 파일 경로 (원본 확장자 유지).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise RuntimeError('huggingface_hub가 필요합니다: pip install -e ".[ml]"') from e

    out_dir = raw_dir / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for split, repo_path in source.files.items():
        cached = hf_hub_download(
            repo_id=source.repo_id,
            filename=repo_path,
            repo_type="dataset",
            revision=source.revision,
        )
        dest = out_dir / f"{split}{Path(repo_path).suffix}"
        shutil.copyfile(cached, dest)
        saved[split] = dest
    return saved


def download_all(
    names: Iterable[str] | None = None, raw_dir: Path = RAW_DIR
) -> dict[str, dict[str, Path]]:
    """이름으로 소스를 골라 내려받는다. ``names``가 비면 전체."""
    by_name = {s.name: s for s in ALL_SOURCES}
    selected = list(names) if names else list(by_name)
    unknown = [n for n in selected if n not in by_name]
    if unknown:
        raise KeyError(f"알 수 없는 소스: {unknown} (가능: {sorted(by_name)})")
    return {n: download_source(by_name[n], raw_dir) for n in selected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="원본 데이터셋 다운로드")
    parser.add_argument("sources", nargs="*", help="소스 이름 (생략 시 전체)")
    args = parser.parse_args(argv)
    result = download_all(args.sources)
    for name, files in result.items():
        for split, path in files.items():
            print(f"{name}/{split}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
