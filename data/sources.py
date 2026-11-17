"""외부 데이터 소스 레지스트리.

각 소스는 HuggingFace Hub의 데이터셋 레포를 특정 커밋(revision)에 고정해
재현 가능한 다운로드를 보장한다. 라이선스 상세는 ``data/LICENSES.md`` 참고.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFSource:
    """HuggingFace Hub 데이터셋 소스 정의.

    Attributes:
        name: 로컬 저장 디렉터리 이름 (``data/raw/<name>/``).
        repo_id: Hub 데이터셋 레포 ID.
        revision: 고정 커밋 SHA (재현성).
        files: 내려받을 레포 내 파일 경로 목록 (split 이름 → 경로).
        license: 데이터셋 카드에 표기된 라이선스 식별자.
        role: 데이터 용도 (``attack`` / ``benign`` / ``mixed``).
        label_note: 원본 라벨 의미 메모 (통합 단계 S4-5에서 사용).
    """

    name: str
    repo_id: str
    revision: str
    files: dict[str, str]
    license: str
    role: str
    label_note: str = ""


# S4-1: 영어 공격 데이터
EN_ATTACK_SOURCES: tuple[HFSource, ...] = (
    HFSource(
        name="deepset_prompt_injections",
        repo_id="deepset/prompt-injections",
        revision="4f61ecb038e9c3fb77e21034b22511b523772cdd",
        files={
            "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
            "test": "data/test-00000-of-00001-701d16158af87368.parquet",
        },
        license="apache-2.0",
        role="mixed",
        label_note="label: 1=injection, 0=benign. 영어·독일어 혼재 (언어 필터는 S4-5).",
    ),
)

# S4-2: 정상 문장 (영어/한국어)
BENIGN_SOURCES: tuple[HFSource, ...] = (
    HFSource(
        name="dolly_15k",
        repo_id="databricks/databricks-dolly-15k",
        revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        files={"train": "databricks-dolly-15k.jsonl"},
        license="cc-by-sa-3.0",
        role="benign",
        label_note="영어. `instruction` 컬럼 사용 (15,011건, 8개 category). 전부 정상(0).",
    ),
    HFSource(
        name="klue_mrc",
        repo_id="klue/klue",
        revision="349481ec73fff722f88e0453ca05c77a447d967c",
        files={
            "train": "mrc/train-00000-of-00001.parquet",
            "validation": "mrc/validation-00000-of-00001.parquet",
        },
        license="cc-by-sa-4.0",
        role="benign",
        label_note="한국어. `question` 컬럼만 사용 (train 17,554 / valid 5,841). 전부 정상(0).",
    ),
)

# 공격 1건당 정상 문장 목표 수 (언어별 적용). 실제 샘플링은 S4-5 통합 단계에서 수행.
BENIGN_PER_ATTACK: int = 2

ALL_SOURCES: tuple[HFSource, ...] = EN_ATTACK_SOURCES + BENIGN_SOURCES
