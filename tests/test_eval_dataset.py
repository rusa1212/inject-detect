"""S5-2 평가 데이터 로더 테스트 (임시 jsonl 사용, 실제 data/processed 불필요)."""

import json

import pytest

from eval.dataset import Sample, load_split


def row(i, split, label=1, lang="ko", category="jailbreak", evasion=None, **extra):
    return {
        "id": f"r{i}",
        "text": f"text {i}",
        "label": label,
        "lang": lang,
        "category": category,
        "source": "test",
        "group": f"g{i}",
        "evasion": evasion,
        "split": split,
        **extra,
    }


@pytest.fixture
def data_dir(tmp_path):
    rows = {
        "train": [row(1, "train"), row(2, "train", label=0, category="benign")],
        "valid": [row(3, "valid", evasion="base64")],
        "test": [
            row(4, "test", lang="en", unknown_field="ignored"),
            row(5, "test", label=0, category="hard_negative"),
        ],
    }
    for split, rs in rows.items():
        lines = [json.dumps(r, ensure_ascii=False) for r in rs]
        (tmp_path / f"{split}.jsonl").write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    return tmp_path


def test_load_single_split(data_dir):
    samples = load_split("test", data_dir)
    assert [s.id for s in samples] == ["r4", "r5"]
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[0].lang == "en" and samples[0].is_original
    assert samples[1].is_hard_negative and not samples[0].is_hard_negative


def test_load_all_splits(data_dir):
    samples = load_split("all", data_dir)
    assert [s.id for s in samples] == ["r1", "r2", "r3", "r4", "r5"]
    assert not samples[2].is_original and samples[2].evasion == "base64"


def test_unknown_split(data_dir):
    with pytest.raises(ValueError, match="split"):
        load_split("dev", data_dir)


def test_missing_file_hints_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.build"):
        load_split("test", tmp_path)


@pytest.mark.parametrize(
    "line",
    ["{not json", json.dumps({"id": "x"}), json.dumps(row(9, "test", label=2))],
)
def test_bad_row(tmp_path, line):
    (tmp_path / "test.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test.jsonl:1"):
        load_split("test", tmp_path)
