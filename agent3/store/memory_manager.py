"""메모리 매니저: 세션 + 장기 SQLite + 런타임 상태(enabled/pending)를 통합.

- 파이프라인 단계들과 서비스가 공유하는 단일 진입점.
- 확인/취소 같은 제어 표현만 규칙 기반(사용자 요구: 일부 미들웨어성 동작 허용).
- 요약본은 지금은 간단 규칙 요약. (추후 LLM 요약으로 교체 가능)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent3.schemas.pipeline import MemoryContext
from agent3.store.db import LongTermStore
from agent3.store.session import SessionStore

CONFIRM_WORDS = {"동의", "확인", "네", "예", "응", "진행", "삭제진행", "그래", "좋아"}
CANCEL_WORDS = {"취소", "중단", "그만", "아니", "아니요", "안해", "안할래", "보류"}

SHORT_INJECT_TURNS = 10           # 컨텍스트 주입 최근 턴
LONG_WINDOW_MESSAGES = 40         # 장기 조회/보관 메시지 수


@dataclass
class RuntimeState:
    enabled: bool = False
    user_code: Optional[str] = None
    pending: Optional[dict[str, Any]] = None
    summary: str = ""


@dataclass
class MemoryManager:
    session: SessionStore = field(default_factory=SessionStore)
    long: LongTermStore = field(default_factory=LongTermStore)
    _runtime: dict[str, RuntimeState] = field(default_factory=dict)

    # ---- identity / runtime ----
    @staticmethod
    def resolve_identity(user_id: str, user_code: Optional[str]) -> str:
        return (user_code or user_id or "").strip()

    def runtime(self, identity: str, user_code: Optional[str] = None) -> RuntimeState:
        state = self._runtime.get(identity)
        if state is None:
            state = RuntimeState(user_code=user_code or identity)
            self._runtime[identity] = state
        elif user_code and not state.user_code:
            state.user_code = user_code
        return state

    # ---- control text ----
    @staticmethod
    def is_confirm(text: str) -> bool:
        return (text or "").strip().replace(" ", "") in CONFIRM_WORDS

    @staticmethod
    def is_cancel(text: str) -> bool:
        return (text or "").strip().replace(" ", "") in CANCEL_WORDS

    # ---- context load ----
    def load_context(self, identity: str, user_code: Optional[str]) -> MemoryContext:
        state = self.runtime(identity, user_code)
        short_window = self.session.get_recent(identity, limit=SHORT_INJECT_TURNS * 2)

        ctx = MemoryContext(
            short_window=short_window,
            short_summary=self._build_summary(short_window, title="단기 요약"),
            long_memory_enabled=state.enabled,
        )

        if state.enabled and state.user_code:
            code = state.user_code
            ctx.long_window = self.long.get_recent_messages(code, limit=LONG_WINDOW_MESSAGES)
            ctx.long_summary = self.long.get_summary(code) or self._build_summary(
                ctx.long_window, title="장기 요약"
            )
            ctx.long_profile = self.long.get_profile(code)
        return ctx

    # ---- persistence (턴 종료 후) ----
    def persist_turn(self, identity: str, query: str, answer: str) -> None:
        self.session.append(identity, "user", query)
        self.session.append(identity, "assistant", answer)

        state = self.runtime(identity)
        if state.enabled and state.user_code:
            code = state.user_code
            self.long.save_message(code, "user", query)
            self.long.save_message(code, "assistant", answer)
            self.long.trim_messages(code, max_messages=LONG_WINDOW_MESSAGES)
            recent = self.long.get_recent_messages(code, limit=LONG_WINDOW_MESSAGES)
            summary = self._build_summary(recent, title="장기 요약")
            state.summary = summary
            self.long.upsert_summary(code, summary)

    # ---- summary (간단 규칙) ----
    @staticmethod
    def _build_summary(history: list[dict[str, str]], *, title: str, max_items: int = 4) -> str:
        if not history:
            return ""
        seen: set[str] = set()
        user_msgs: list[str] = []
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip().replace("\n", " ")
            if not content or content in seen:
                continue
            seen.add(content)
            user_msgs.append(content if len(content) <= 90 else content[:90] + "...")
            if len(user_msgs) >= max_items:
                break
        if not user_msgs:
            return ""
        lines = [title]
        for i, m in enumerate(reversed(user_msgs), start=1):
            lines.append(f"{i}. {m}")
        return "\n".join(lines)
