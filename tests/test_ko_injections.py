"""한국어 인젝션 데이터(S4-3) 검증 테스트."""

from __future__ import annotations

import textwrap

import pytest

from data.korean.dataset import (
    CATEGORIES,
    KoDatasetError,
    load_ko_injections,
    source_id,
    stats,
)

ROWS = load_ko_injections()


def test_target_quantity():
    assert 500 <= len(ROWS) <= 1000


def test_all_categories_covered_with_minimum():
    counts = stats(ROWS)["category"]
    assert set(counts) == CATEGORIES
    assert min(counts.values()) >= 80


def test_all_methods_and_styles_present():
    s = stats(ROWS)
    assert set(s["method"]) == {"translation", "variant", "original"}
    assert set(s["style"]) == {"honorific", "plain", "colloquial"}


def test_ids_unique_and_labels():
    assert len({r.id for r in ROWS}) == len(ROWS)
    assert all(r.label == 1 and r.lang == "ko" for r in ROWS)


def test_translation_rows_have_source():
    for r in ROWS:
        if r.method == "translation":
            assert r.source_id
        if r.method == "original":
            assert r.source_id is None


def test_group_shares_category_and_source():
    by_group: dict[str, set] = {}
    for r in ROWS:
        by_group.setdefault(r.group, set()).add((r.category, r.source_id))
    assert all(len(v) == 1 for v in by_group.values())


def test_source_id_is_sha1_prefix():
    assert source_id("abc") == "a9993e364706"


def _write(tmp_path, body: str):
    (tmp_path / "x.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "body",
    [
        # 잘못된 카테고리
        """
        version: 1
        groups:
          - {id: o1, source: null, category: nope, items: [[original, plain, "무시해"]]}
        """,
        # 한글 없음
        """
        version: 1
        groups:
          - {id: o1, source: null, category: jailbreak, items: [[original, plain, "ignore"]]}
        """,
        # 번역 그룹에 source 없음
        """
        version: 1
        groups:
          - {id: t1, source: null, category: jailbreak, items: [[translation, plain, "무시해"]]}
        """,
        # 직접 작성 그룹에 source 있음
        """
        version: 1
        groups:
          - {id: o1, source: aaaaaaaaaaaa, category: jailbreak,
             items: [[original, plain, "무시해"]]}
        """,
        # 공백만 다른 중복 텍스트
        """
        version: 1
        groups:
          - {id: o1, source: null, category: jailbreak, items: [[original, plain, "다 무시해"]]}
          - {id: o2, source: null, category: jailbreak, items: [[original, plain, "다  무시해 "]]}
        """,
        # 중복 그룹 id
        """
        version: 1
        groups:
          - {id: o1, source: null, category: jailbreak, items: [[original, plain, "가"]]}
          - {id: o1, source: null, category: jailbreak, items: [[original, plain, "나"]]}
        """,
    ],
)
def test_validation_rejects_bad_data(tmp_path, body):
    with pytest.raises(KoDatasetError):
        load_ko_injections(_write(tmp_path, body))
