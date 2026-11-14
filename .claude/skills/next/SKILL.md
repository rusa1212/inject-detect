---
name: next
description: AGENT.md 규칙에 따라 다음 세그먼트(또는 인자로 지정한 세그먼트 ID) 하나를 작업하고 승인을 요청한다. 예) /next, /next S2-1
---

AGENT.md의 작업 사이클을 그대로 수행한다.

1. `docs/PROGRESS.md`를 읽는다.
   - `review` 상태 세그먼트가 있으면 새 작업을 시작하지 말고, 그 세그먼트의 완료 보고를 다시 보여주고 승인부터 받는다.
2. 대상 세그먼트를 정한다: 인자 `$ARGUMENTS`가 있으면 그 ID, 없으면 선행 조건을 만족하는 첫 `todo`.
   지정된 세그먼트의 선행 조건이 미완료면 진행 전에 사용자에게 알리고 확인받는다.
3. `docs/inject-detectWorkPlan.md`에서 해당 세그먼트 체크리스트를 읽고, 구현 → 검증한다.
4. `PROGRESS.md` 상태를 `review`로 바꾸고, AGENT.md의 보고 형식으로 보고한 뒤 AskUserQuestion으로 승인을 묻고 **멈춘다**.
5. 승인되면 AGENT.md "승인 후 마무리" 절차(체크박스·PROGRESS 갱신 → 커밋)를 수행한다.
   "커밋 후 다음 세그먼트"가 선택되면 1번부터 다시 반복한다.
