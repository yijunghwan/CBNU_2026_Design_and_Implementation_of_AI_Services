"""오케스트레이터: 단계 순서와 분기/루프 제어만 담당.

흐름
  security → context_load → planner → memory → slots
          → [tools → responder → judge]  (judge=retry면 제한 루프)
          → output

단락(short-circuit)
  - 보안 차단: 이후 단계 생략하고 즉시 출력
  - 메모리 명령 처리 완료: 이후 단계 생략하고 즉시 출력
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import JudgeDecision, PipelineContext

logger = logging.getLogger(__name__)

MAX_JUDGE_RETRIES = 1


@dataclass
class Orchestrator:
    """단계 리스트를 순서대로 실행하는 실행기."""

    security: Stage
    context_load: Stage
    planner: Stage
    memory: Stage
    slots: Stage
    tools: Stage
    responder: Stage
    judge: Stage
    max_retries: int = MAX_JUDGE_RETRIES

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # 1) 보안 검사 (규칙 기반) — 차단 시 조기 종료
        ctx = self.security(ctx)
        if not ctx.security.passed:
            ctx.final_answer = ctx.security.blocked_reason or "요청을 처리할 수 없습니다."
            return ctx

        # 2) 컨텍스트 로드 (윈도우/요약/프로필)
        ctx = self.context_load(ctx)

        # 3) 플래너 (작은 LLM: 의도/메모리/툴계획/슬롯)
        ctx = self.planner(ctx)

        # 4) 메모리 FSM — 명령/확인 처리 시 조기 종료
        ctx = self.memory(ctx)
        if ctx.memory_result.handled:
            ctx.final_answer = ctx.memory_result.response_text
            return ctx

        # 5) 슬롯 정규화 (맵핑 테이블, 결정론적)
        ctx = self.slots(ctx)

        # 6) 툴 → 응답 → 평가 (제한 루프)
        ctx = self._tool_answer_loop(ctx)
        ctx.final_answer = ctx.draft_answer
        return ctx

    def _tool_answer_loop(self, ctx: PipelineContext) -> PipelineContext:
        while True:
            ctx = self.tools(ctx)
            ctx = self.responder(ctx)
            ctx = self.judge(ctx)

            decision = ctx.judge.decision
            if decision == JudgeDecision.RETRY.value and ctx.retry_count < self.max_retries:
                ctx.retry_count += 1
                ctx.debug["retry_refined_query"] = ctx.judge.refined_query
                logger.info("judge requested retry (%d/%d)", ctx.retry_count, self.max_retries)
                continue

            if decision == JudgeDecision.CLARIFY.value:
                ctx.draft_answer = ctx.judge.clarifying_question or ctx.draft_answer
            return ctx


def build_default_pipeline(manager=None) -> Orchestrator:
    """실제 단계 구현을 조립. manager 미지정 시 기본 MemoryManager 생성."""
    from agent3.pipeline.stages.security import SecurityStage
    from agent3.pipeline.stages.context_load import ContextLoadStage
    from agent3.pipeline.stages.planner import PlannerStage
    from agent3.pipeline.stages.memory import MemoryStage
    from agent3.pipeline.stages.slots import SlotStage
    from agent3.pipeline.stages.tools import ToolStage
    from agent3.pipeline.stages.responder import ResponderStage
    from agent3.pipeline.stages.judge import JudgeStage
    from agent3.store.memory_manager import MemoryManager

    manager = manager or MemoryManager()

    return Orchestrator(
        security=SecurityStage(),
        context_load=ContextLoadStage(manager),
        planner=PlannerStage(),
        memory=MemoryStage(manager),
        slots=SlotStage(),
        tools=ToolStage(),
        responder=ResponderStage(),
        judge=JudgeStage(),
    )
