"""data.sources / data.download 테스트 (네트워크 없이 실행)."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from data import download
from data.sources import ALL_SOURCES, EN_ATTACK_SOURCES

LICENSES_MD = Path(__file__).resolve().parents[1] / "data" / "LICENSES.md"


def test_source_names_unique():
    names = [s.name for s in ALL_SOURCES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("source", ALL_SOURCES, ids=lambda s: s.name)
def test_source_is_pinned_and_licensed(source):
    assert re.fullmatch(r"[0-9a-f]{40}", source.revision), "revision은 커밋 SHA로 고정"
    assert source.license
    assert source.files
    assert source.role in {"attack", "benign", "mixed"}


@pytest.mark.parametrize("source", ALL_SOURCES, ids=lambda s: s.name)
def test_source_documented_in_licenses(source):
    assert source.repo_id in LICENSES_MD.read_text(encoding="utf-8")


def test_en_attack_sources_included():
    assert all(s in ALL_SOURCES for s in EN_ATTACK_SOURCES)
    assert any(s.repo_id == "deepset/prompt-injections" for s in EN_ATTACK_SOURCES)


def test_download_source_copies_pinned_files(tmp_path, monkeypatch):
    calls = []

    def fake_hf_hub_download(repo_id, filename, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        p = tmp_path / "cache" / filename.replace("/", "_")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PAR1")
        return str(p)

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    source = EN_ATTACK_SOURCES[0]
    saved = download.download_source(source, raw_dir=tmp_path / "raw")

    assert set(saved) == set(source.files)
    for split, path in saved.items():
        assert path == tmp_path / "raw" / source.name / f"{split}.parquet"
        assert path.read_bytes() == b"PAR1"
    assert all(c[2] == "dataset" and c[3] == source.revision for c in calls)


def test_download_all_rejects_unknown(tmp_path):
    with pytest.raises(KeyError):
        download.download_all(["nope"], raw_dir=tmp_path)
