"""컨텍스트 로드 단계: 단기/장기 윈도우·요약·프로필을 채운다.

- 단기: 세션 저장소에서 최근 대화(주입은 컨텍스트 빌더가 10턴으로 제한).
- 장기: 장기기억 ON일 때 DB에서 최근 대화 + 요약 + 프로필 로드.
"""

from __future__ import annotations

from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import PipelineContext
from agent3.store.memory_manager import MemoryManager


class ContextLoadStage(Stage):
    name = "context_load"

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def run(self, ctx: PipelineContext) -> PipelineContext:
        identity = self.manager.resolve_identity(ctx.user_id, ctx.user_code)
        ctx.memory = self.manager.load_context(identity, ctx.user_code)
        return ctx
