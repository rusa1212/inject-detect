# promptshield-ko 데이터 카드

한국어·영어 프롬프트 인젝션 탐지기의 학습·평가용 데이터셋.
**레포에는 원본 데이터와 빌드 산출물이 포함되지 않는다.** 아래 명령 한 번으로 전체를 재생성한다.

```bash
pip install -e ".[data]"    # huggingface_hub·pandas·pyarrow·numpy (다운로드·빌드용)
python -m data.build        # 원본 다운로드(없을 때만) → 통합 → 분할 → data/processed/
```

같은 코드·같은 원본 revision이면 산출물은 바이트 단위로 동일하다 (모든 샘플링·분할이 해시 기반으로 결정적).

## 1. 구성

| 구분 | 출처 | 언어 | 라벨 | 레포 포함 | 세그먼트 |
|---|---|---|---|---|---|
| 영어 공격·정상 | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (영어 문장만) | en | 0/1 | ✗ (다운로드) | S4-1 |
| 영어 일반 정상 | [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) `instruction` | en | 0 | ✗ (다운로드) | S4-2 |
| 한국어 일반 정상 | [KLUE MRC](https://huggingface.co/datasets/klue/klue) `question` | ko | 0 | ✗ (다운로드) | S4-2 |
| **한국어 인젝션** | 직접 구축: deepset 번역·변형 + 직접 작성 ([가이드라인](korean/GUIDELINES.md)) | ko | 1 | ✓ `korean/injections/` | S4-3 |
| **하드 네거티브** | 직접 작성: 인젝션 키워드가 들어간 정상 문장 | ko·en | 0 | ✓ `hard_negatives/` | S4-4 |
| **우회 변형** | 공격 문장에 Base64·동형문자·자모 분리·구분자 삽입 자동 적용 | ko·en | 1 | ✗ (생성) | S4-4 |

라이선스와 주의 사항은 [LICENSES.md](LICENSES.md) 참고. CC BY-SA 소스(dolly, KLUE)가 포함되므로
빌드 산출물을 배포할 때는 **CC BY-SA 4.0**으로 배포하고 모든 출처를 표기한다.

## 2. 스키마

`data/processed/{train,valid,test}.jsonl`, 한 줄에 한 건:

| 필드 | 설명 |
|---|---|
| `text` | 입력 문장 |
| `label` | 1=공격(인젝션), 0=정상 |
| `lang` | `ko` / `en` |
| `category` | 공격: `instruction_override` · `role_hijack` · `jailbreak` · `prompt_leak` (한국어), `unknown` (deepset 영어: 원본에 카테고리 없음) / 정상: `benign`, `hard_negative` |
| `source` | `ko_injection` · `hard_negative` · `deepset` · `dolly_15k` · `klue_mrc` (우회 변형은 원문의 source) |
| `split` | `train` / `valid` / `test` |
| `id` | 추적용 ID (`t001-2`, `deepset-<sha1 12자리>`, `<원문 id>~jamo` 등) |
| `group` | 누수 방지 단위. 같은 group은 항상 같은 split |
| `evasion` | 우회 변형 종류 (`base64`·`homoglyph`·`jamo`·`spacing`), 원문이면 `null` |

`stats.json`에 분할·언어·라벨·카테고리·출처별 건수와 중복 제거 결과가 기록된다.

## 3. 구축 방법 (`data/build.py`)

1. **수집**
   - deepset에서 **영어 문장만** 남긴다. 불용어·움라우트 휴리스틱으로 독일어와 영·독 혼합 문장을 제외한다
     (662 → 420건). 원본 train/test 구분은 쓰지 않고 다시 분할한다.
   - 한국어 인젝션 500건, 하드 네거티브 204건(ko 144·en 60)을 YAML에서 읽는다.
2. **중복 제거**
   - 완전 중복: casefold 후 구두점·공백을 지운 키가 같으면 하나만 남긴다
     (우선순위: 한국어 인젝션 > 하드 네거티브 > deepset > dolly > KLUE). 라벨이 엇갈리면 양쪽 다 버린다.
   - 근접 중복: 문자 5-gram MinHash(64 perm, 16 band) LSH로 후보를 찾고,
     Jaccard ≥ 0.85이면서 라벨이 같으면 버린다.
   - 포함 관계: 짧은 문장의 5-gram 90% 이상이 긴 문장에 들어 있으면 두 group을 합친다.
     deepset의 '정상 질문 + 공격' 이어붙이기 문장이 원래 공격 문장과 다른 split에 들어가는 것을 막는다.
3. **정상 샘플링**: 언어별로 *원문 공격 1건당 일반 정상 2건*(`data.sources.BENIGN_PER_ATTACK`)을 맞춘다.
   텍스트 해시 순으로 결정적으로 고른다. 영어는 deepset 정상을 먼저 세고 모자란 만큼 dolly로 채운다.
   하드 네거티브는 이 비율과 별개로 전부 넣는다.
4. **우회 변형**: 중복 제거 후 남은 공격 원문 전체에 S4-4 생성기를 적용한다. 변형은 원문의 group을 물려받는다.
5. **분할**: `sha1(group)` 값으로 train/valid/test = 80/10/10을 정한다. group은 다음을 하나로 묶는다:
   - 한국어 번역 그룹은 영어 원문 ID를 group으로 쓴다 → **원문-번역 쌍이 같은 split에 배치**된다.
   - 같은 그룹 안의 존댓말·반말·구어체 변형
   - 원문과 그 우회 변형
   - 포함 관계로 합친 deepset 문장들

## 4. 통계 (2026-11-17 빌드 기준)

총 **4,210건** / group 1,891개. 중복 제거 결과: 완전 중복 9건·근접 중복 4건 삭제, group 병합 51건, 라벨 충돌 0건.

| split | lang | 공격(원문) | 일반 정상 | 하드 네거티브 | 우회 변형(공격) | 합계 |
|---|---|---|---|---|---|---|
| train | ko | 383 | 817 | 114 | 1,177 | 2,491 |
| train | en | 123 | 264 | 45 | 367 | 799 |
| valid | ko | 47 | 90 | 19 | 145 | 301 |
| valid | en | 18 | 32 | 7 | 54 | 111 |
| test | ko | 70 | 93 | 11 | 215 | 389 |
| test | en | 21 | 27 | 8 | 63 | 119 |
| **합계** | | **662** | **1,323** | **204** | **2,021** | **4,210** |

우회 변형 내역: ko base64 500 · jamo 498 · spacing 498 · homoglyph 41 / en base64 162 · homoglyph 161 · spacing 161.

한국어 공격 카테고리(변형 포함): instruction_override 712 · prompt_leak 460 · jailbreak 445 · role_hijack 420.

> 원문 기준 공격:정상 비율은 1:2다. 우회 변형까지 넣으면 공격 쪽이 더 많아진다(ko 2,037 vs 1,144).
> 학습·평가 목적에 따라 `evasion`이 null인 행만 쓰거나 변형을 따로 평가하는 식으로 골라 쓴다.

## 5. 알려진 한계

- **규모**: 영어 test 세트가 작다(원문 공격 21건). 영어 지표는 신뢰구간이 넓으니 해석에 주의한다.
  group 해시 분할이라 split 비율도 정확히 80/10/10은 아니다.
- **한국어 공격의 출처**: 실제 사용자 로그가 아니라 에이전트 초안 + 관리자 승인으로 만들었다. 표본 검토는 아직 하지 않았다.
  작성자가 한 명이라 문체 다양성에 한계가 있다.
- **정상 문장 분포**: 한국어 일반 정상 문장은 KLUE 독해 질문("~는 무엇인가?")이라 실제 챗봇 대화체와 다르다.
  모델이 "문체 = 라벨"이라는 지름길을 배울 위험이 있어, 대화체 하드 네거티브로 일부 보완했다.
- **deepset 편향**: 독일 정치·언론 주제, 짧은 지시형 문장에 치우쳐 있다. 영어 공격에는 카테고리 라벨이 없다(`unknown`).
- **언어 판정 휴리스틱**: 불용어 기반이라 스페인어 등 소수 비영어 문장이 `en`으로 남았을 수 있다.
- **우회 변형**: 기계적으로 만든 것이라 실제 공격자의 변형보다 규칙적이다. Base64 템플릿은 3종뿐이다.
- **누수 방지 범위**: 근접 중복·포함 관계는 문자 5-gram 기준이다. 의미는 같지만 표현이 완전히 다른 문장(예: 같은 공격의 다른 번역)은
  group 정보가 없으면 서로 다른 split에 들어갈 수 있다.
