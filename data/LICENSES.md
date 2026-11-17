# 데이터셋 라이선스

레포에는 원본 데이터를 포함하지 않으며, `python -m data.download`로 각 소스를 직접 내려받는다.
아래 라이선스는 기록일 기준 데이터셋 카드 표기이며, 재배포·상업적 이용 전 원 출처를 다시 확인할 것.

| 소스 | 용도 | 고정 revision | 라이선스 | 확인일 | 비고 |
|---|---|---|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 영어 공격 + 정상 (S4-1) | `4f61ecb` | Apache-2.0 (카드 메타데이터 `license`) | 2026-11-17 | 아래 주의 참고 |

## 주의 사항

### deepset/prompt-injections
- 카드 YAML의 최상위 `license`는 `apache-2.0`이지만, `dataset_info` 블록 안에는 `license: cc-by-4.0`이
  함께 적혀 있어 표기가 일치하지 않는다. 두 라이선스 모두 출처 표기 조건으로 재사용을 허용하므로,
  **출처(deepset)를 명시**하고 파생 데이터 배포 시 두 조건을 모두 충족하도록 한다.
- 규모: train 546 / test 116 (총 662). `label` 1=injection, 0=benign.
- 영어와 독일어 문장이 섞여 있음 → 언어 필터링은 S4-5 통합 단계에서 처리.
