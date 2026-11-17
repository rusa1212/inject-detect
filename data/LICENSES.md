# 데이터셋 라이선스

레포에는 원본 데이터를 포함하지 않으며, `python -m data.download`로 각 소스를 직접 내려받는다.
아래 라이선스는 기록일 기준 데이터셋 카드 표기이며, 재배포·상업적 이용 전 원 출처를 다시 확인할 것.

| 소스 | 용도 | 고정 revision | 라이선스 | 확인일 | 비고 |
|---|---|---|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 영어 공격 + 정상 (S4-1) | `4f61ecb` | Apache-2.0 (카드 메타데이터 `license`) | 2026-11-17 | 아래 주의 참고 |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 영어 정상 (S4-2) | `bdd27f4` | CC-BY-SA-3.0 | 2026-11-17 | 아래 주의 참고 |
| [klue/klue](https://huggingface.co/datasets/klue/klue) (`mrc` 설정) | 한국어 정상 (S4-2) | `349481e` | CC-BY-SA-4.0 (카드 메타데이터 `license`) | 2026-11-17 | 아래 주의 참고 |

## 주의 사항

### deepset/prompt-injections
- 카드 YAML의 최상위 `license`는 `apache-2.0`이지만, `dataset_info` 블록 안에는 `license: cc-by-4.0`이
  함께 적혀 있어 표기가 일치하지 않는다. 두 라이선스 모두 출처 표기 조건으로 재사용을 허용하므로,
  **출처(deepset)를 명시**하고 파생 데이터 배포 시 두 조건을 모두 충족하도록 한다.
- 규모: train 546 / test 116 (총 662). `label` 1=injection, 0=benign.
- 영어와 독일어 문장이 섞여 있음 → 언어 필터링은 S4-5 통합 단계에서 처리.

### databricks/databricks-dolly-15k
- Databricks 임직원이 작성한 지시문·응답 데이터. 카드 본문에 CC BY-SA 3.0 명시
  (일부 카테고리는 Wikipedia 등 CC BY-SA 3.0 자료 인용 포함).
- **ShareAlike 조건**: 이 데이터를 포함·가공한 파생 데이터셋을 배포할 경우 같은 라이선스(CC BY-SA)로
  배포해야 하고 출처(Databricks)를 표기해야 한다. 코드(MIT)와 데이터 라이선스는 별개로 관리한다.
- 사용 컬럼: `instruction` (15,011건). `context`·`response`는 사용하지 않는다.

### klue/klue (mrc)
- KLUE 벤치마크의 기계독해(MRC) 데이터. 카드 YAML 메타데이터는 `cc-by-sa-4.0`이지만
  카드 본문의 "Licensing Information" 항목은 "Needs More Information"으로 비어 있다.
  메타데이터 표기(CC BY-SA 4.0)를 따르고 출처(KLUE, Park et al., 2021)를 표기한다.
- **ShareAlike 조건**: 파생 데이터셋 배포 시 CC BY-SA 적용 (dolly와 동일).
- 사용 컬럼: `question` (train 17,554 / validation 5,841). 지문(`context`)은 뉴스(한국경제·아크로팬)와
  위키백과 출처이므로 사용하지 않는다. 질문은 독해 문체("~는 무엇인가?")라 대화체 정상 문장으로는 한계가 있다.

## 파생 데이터 배포 시 요약
- S4-5에서 만든 통합 데이터셋을 외부에 배포하면(S12-2 HuggingFace Hub 등) CC BY-SA 소스가 포함되므로
  **데이터셋 전체를 CC BY-SA 4.0으로 배포**하고 위 출처를 모두 표기한다.
