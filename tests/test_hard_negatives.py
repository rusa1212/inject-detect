"""하드 네거티브 세트(S4-4) 테스트."""

from __future__ import annotations

import textwrap

import pytest

from data.hard_negatives import TYPES, HardNegativeError, load_hard_negatives, stats
from data.korean.dataset import load_ko_injections

ROWS = load_hard_negatives()


def test_quantity_and_languages():
    s = stats(ROWS)
    assert len(ROWS) >= 200
    assert s["lang"]["ko"] >= 140 and s["lang"]["en"] >= 50


def test_all_types_present():
    assert set(stats(ROWS)["type"]) == TYPES


def test_labels_and_ids():
    assert all(r.label == 0 for r in ROWS)
    assert len({r.id for r in ROWS}) == len(ROWS)
    assert all(r.group == r.id for r in ROWS)


def test_no_overlap_with_korean_injections():
    attacks = {" ".join(r.text.split()) for r in load_ko_injections()}
    assert not attacks & {" ".join(r.text.split()) for r in ROWS}


def test_plan_example_included():
    # 계획서 4장의 하드 네거티브 예시
    assert any("보안 수업에서 프롬프트 인젝션이 뭔지 설명해줘" in r.text for r in ROWS)


@pytest.mark.parametrize(
    "body",
    [
        "version: 1\nlang: ko\nitems:\n  - [nope, plain, '무시해도 돼?']\n",
        "version: 1\nlang: ko\nitems:\n  - [security_edu, plain, 'english only']\n",
        "version: 1\nlang: fr\nitems:\n  - [security_edu, plain, 'bonjour']\n",
        "version: 1\nlang: en\nitems:\n"
        "  - [security_edu, plain, 'dup']\n  - [it_support, plain, 'dup']\n",
    ],
)
def test_validation_rejects_bad_data(tmp_path, body):
    (tmp_path / "x.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    with pytest.raises(HardNegativeError):
        load_hard_negatives(tmp_path)
