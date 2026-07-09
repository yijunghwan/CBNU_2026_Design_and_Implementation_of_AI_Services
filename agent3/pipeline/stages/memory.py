"""메모리 FSM 단계: 장기기억 켜기/끄기, 프로필 수정, 삭제(선택), 조회.

동작 원칙
- 개인정보 저장(enable)/수정/삭제는 확인(동의/취소) 단계를 거친다.
- disable / show_profile 은 즉시 처리.
- 확인/취소 같은 제어 표현만 규칙 기반, 나머지 의도는 플래너(LLM)가 memory_action으로 전달.
"""

from __future__ import annotations

from agent3.mapping import SlotNormalizer
from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import MemoryAction, MemoryResult, PipelineContext
from agent3.store.memory_manager import MemoryManager

_CONFIRM_REQUIRED = {
    MemoryAction.ENABLE.value,
    MemoryAction.UPDATE_PROFILE.value,
    MemoryAction.DELETE_PROFILE.value,
    MemoryAction.DELETE_HISTORY.value,
    MemoryAction.DELETE_ALL.value,
}

_ACTION_LABEL = {
    MemoryAction.ENABLE.value: "장기기억 사용",
    MemoryAction.UPDATE_PROFILE.value: "개인정보 변경",
    MemoryAction.DELETE_PROFILE.value: "개인정보 삭제",
    MemoryAction.DELETE_HISTORY.value: "대화기록 삭제",
    MemoryAction.DELETE_ALL.value: "장기기억 전체 삭제",
}

_PROFILE_KEYS = [
    "age", "region", "income", "employment_status",
    "marriage_status", "children_count", "housing_status",
]


class MemoryStage(Stage):
    name = "memory"

    def __init__(self, manager: MemoryManager):
        self.manager = manager
        self.normalizer = SlotNormalizer()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        identity = self.manager.resolve_identity(ctx.user_id, ctx.user_code)
        state = self.manager.runtime(identity, ctx.user_code)
        query = ctx.query
        action = ctx.plan.memory_action

        # 1) 대기 중 확인/취소 우선 처리
        if state.pending:
            if self.manager.is_confirm(query):
                msg = self._execute(state.pending["action"], state, state.pending.get("payload", query))
                done = state.pending["action"]
                state.pending = None
                ctx.memory_result = MemoryResult(handled=True, response_text=msg, action=done, status="completed")
                return ctx
            if self.manager.is_cancel(query):
                canceled = state.pending["action"]
                state.pending = None
                ctx.memory_result = MemoryResult(
                    handled=True, response_text="대기 중인 요청을 취소했어요.", action=canceled, status="canceled"
                )
                return ctx
            ctx.memory_result = MemoryResult(
                handled=True,
                response_text="진행하려면 '동의', 취소하려면 '취소'라고 답해주세요.",
                action=state.pending["action"],
                status="needs_confirmation",
                requires_confirmation=True,
            )
            return ctx

        # 2) 신규 메모리 액션
        if action == MemoryAction.NONE.value:
            ctx.memory_result = MemoryResult(handled=False)
            return ctx

        if action == MemoryAction.DISABLE.value:
            state.enabled = False
            ctx.memory_result = MemoryResult(
                handled=True, response_text="장기기억을 해제했어요. 지금부터 저장/조회하지 않습니다.",
                action=action, status="completed",
            )
            return ctx

        if action == MemoryAction.SHOW_PROFILE.value:
            ctx.memory_result = MemoryResult(
                handled=True, response_text=self._profile_snapshot(state.user_code),
                action=action, status="completed",
            )
            return ctx

        if action == MemoryAction.ENABLE.value and not ctx.user_code:
            ctx.memory_result = MemoryResult(
                handled=True,
                response_text="장기기억을 이어 쓰려면 유저코드가 필요해요. 유저코드를 입력한 뒤 다시 '장기기억 켜줘'라고 해주세요.",
                action=action, status="needs_user_code",
            )
            return ctx

        # 수정/삭제는 장기기억 ON 상태에서만
        if action in {MemoryAction.UPDATE_PROFILE.value, MemoryAction.DELETE_PROFILE.value,
                      MemoryAction.DELETE_HISTORY.value, MemoryAction.DELETE_ALL.value} and not state.enabled:
            ctx.memory_result = MemoryResult(
                handled=True,
                response_text="먼저 '장기기억 켜줘'로 장기기억을 활성화해 주세요.",
                action=action, status="needs_enable",
            )
            return ctx

        # 3) 확인 필요 → pending 등록
        if action in _CONFIRM_REQUIRED:
            state.pending = {"action": action, "payload": query}
            label = _ACTION_LABEL.get(action, "요청")
            ctx.memory_result = MemoryResult(
                handled=True,
                response_text=f"'{label}'을(를) 진행할까요? 개인정보 처리 동의를 위해 '동의' 또는 '확인'이라고 답해주세요.",
                action=action, status="needs_confirmation", requires_confirmation=True,
            )
            return ctx

        ctx.memory_result = MemoryResult(handled=False)
        return ctx

    # ---- 실제 실행 ----
    def _execute(self, action: str, state, payload: str) -> str:
        code = state.user_code
        if action == MemoryAction.ENABLE.value:
            self.manager.long.ensure_user(code)
            state.enabled = True
            return (
                "장기기억을 켰어요. 이제 대화와 (알려주시는) 개인정보를 저장해 맞춤 추천에 활용합니다.\n"
                "- 개인정보 변경: '개인정보 변경: 만 26세, 서울, 취업준비중'\n"
                "- 끄기: '장기기억 꺼줘'"
            )

        if action == MemoryAction.UPDATE_PROFILE.value:
            fields = self._extract_profile(payload)
            if not fields:
                return "변경할 개인정보를 찾지 못했어요. 예: '개인정보 변경: 만 27세, 부산, 취업준비중, 미혼, 무주택'"
            self.manager.long.update_profile(code, fields)
            return "개인정보를 반영했어요."

        if action == MemoryAction.DELETE_PROFILE.value:
            self.manager.long.clear_profile(code)
            return "개인정보(프로필)만 삭제했어요. 대화기록은 유지됩니다."

        if action == MemoryAction.DELETE_HISTORY.value:
            n = self.manager.long.delete_messages(code)
            return f"장기 대화기록을 삭제했어요. (삭제 {n}건)"

        if action == MemoryAction.DELETE_ALL.value:
            self.manager.long.delete_user(code)
            self.manager.long.ensure_user(code)
            return "장기기억 전체(개인정보+대화기록)를 삭제했어요. 장기기억 모드는 계속 켜져 있어요."

        return "요청을 처리하지 못했어요."

    def _extract_profile(self, text: str) -> dict:
        slots = self.normalizer.from_text(text)
        return {k: getattr(slots, k) for k in _PROFILE_KEYS if getattr(slots, k) not in (None, "", [])}

    def _profile_snapshot(self, user_code) -> str:
        if not user_code:
            return "저장된 개인정보가 없어요."
        profile = self.manager.long.get_profile(user_code)
        if not profile:
            return "저장된 개인정보가 없어요. '개인정보 변경: ...' 형식으로 알려주세요."
        ko = {
            "age": "나이", "region": "지역", "income": "월소득",
            "employment_status": "고용상태", "marriage_status": "결혼여부",
            "children_count": "자녀수", "housing_status": "주거상태",
        }
        lines = ["현재 저장된 개인정보예요."]
        for key, label in ko.items():
            if key in profile:
                lines.append(f"- {label}: {profile[key]}")
        return "\n".join(lines)
