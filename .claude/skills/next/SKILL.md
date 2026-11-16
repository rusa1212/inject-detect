---
name: next
description: AGENT.md 규칙에 따라 다음 세그먼트(또는 인자로 지정한 세그먼트 ID) 하나를 작업하고 승인을 요청한다. 커밋·푸시는 하지 않는다. 예) /next, /next S2-1
---

AGENT.md의 작업 사이클을 그대로 수행한다. **git commit/push/add는 절대 실행하지 않는다.**

1. `git status`로 미커밋 변경을 확인한다. 있으면 사용자에게 먼저 커밋해달라고 요청하고 멈춘다
   (사용자가 "그냥 진행"이라고 하면 계속).
2. `docs/PROGRESS.md`를 읽는다.
   - `review` 상태 세그먼트가 있으면 새 작업을 시작하지 말고, 그 세그먼트의 완료 보고를 다시 보여주고 승인부터 받는다.
3. 대상 세그먼트를 정한다: 인자 `$ARGUMENTS`가 있으면 그 ID, 없으면 선행 조건을 만족하는 첫 `todo`.
   지정된 세그먼트의 선행 조건이 미완료면 진행 전에 사용자에게 알리고 확인받는다.
4. `docs/inject-detectWorkPlan.md`에서 해당 세그먼트 체크리스트를 읽고, 구현 → 검증한다.
5. `PROGRESS.md` 상태를 `review`로 바꾸고, AGENT.md의 보고 형식으로 보고한 뒤 AskUserQuestion으로 승인을 묻고 **멈춘다**.
6. 승인되면 체크박스·PROGRESS를 갱신하고, `git status`로 변경·추가된 파일을 확인한 뒤
   AGENT.md 5)의 형식대로 **파일별** `git add` + 커밋 메시지 제안을 의존 순서대로 출력하고 멈춘다.
