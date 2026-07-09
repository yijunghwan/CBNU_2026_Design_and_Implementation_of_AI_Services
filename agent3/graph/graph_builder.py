"""agent3: LangGraph StateGraph 기반 오케스트레이션.

설계 원칙
- 기존 로직(=agent2의 각 Stage 구현)을 그대로 재사용한다.
- 단계 간 순서/분기/루프 제어만 LangGraph StateGraph로 표현한다.
- 상태는 PipelineContext 하나를 그래프 전 구간에서 공유한다(기존과 동일한 계약).

기존 오케스트레이터와 동일한 흐름
  security → context_load → planner → memory → slots
          → [tools → responder → judge]  (judge=retry면 제한 루프)
          → output

조건부 분기(conditional edge)
  - security: 차단 → finalize_blocked / 통과 → context_load
  - memory:   명령 처리 완료 → finalize_memory / 통과 → slots
  - judge:    retry(상한 이내) → retry_prep → tools / 그 외 → finalize_answer
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

# 기존 로직 재사용 (agent2 Stage 구현을 그대로 노드로 사용)
from agent3.pipeline.stages.context_load import ContextLoadStage
from agent3.pipeline.stages.judge import JudgeStage
from agent3.pipeline.stages.memory import MemoryStage
from agent3.pipeline.stages.planner import PlannerStage
from agent3.pipeline.stages.responder import ResponderStage
from agent3.pipeline.stages.security import SecurityStage
from agent3.pipeline.stages.slots import SlotStage
from agent3.pipeline.stages.tools import ToolStage
from agent3.schemas.pipeline import JudgeDecision, PipelineContext
from agent3.store.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

MAX_JUDGE_RETRIES = 1


class GraphState(TypedDict):
    """그래프 상태: PipelineContext 하나만 공유한다(기존 계약 유지)."""

    ctx: PipelineContext


def build_graph(manager: MemoryManager | None = None):
    """LangGraph StateGraph를 조립하고 컴파일한다.

    각 노드는 agent2의 Stage 인스턴스를 그대로 호출하므로 로직은 변하지 않는다.
    """
    manager = manager or MemoryManager()

    security = SecurityStage()
    context_load = ContextLoadStage(manager)
    planner = PlannerStage()
    memory = MemoryStage(manager)
    slots = SlotStage()
    tools = ToolStage()
    responder = ResponderStage()
    judge = JudgeStage()

    # ---- 단계 노드 (기존 Stage를 그대로 실행) ----
    def n_security(state: GraphState) -> GraphState:
        return {"ctx": security(state["ctx"])}

    def n_context_load(state: GraphState) -> GraphState:
        return {"ctx": context_load(state["ctx"])}

    def n_planner(state: GraphState) -> GraphState:
        return {"ctx": planner(state["ctx"])}

    def n_memory(state: GraphState) -> GraphState:
        return {"ctx": memory(state["ctx"])}

    def n_slots(state: GraphState) -> GraphState:
        return {"ctx": slots(state["ctx"])}

    def n_tools(state: GraphState) -> GraphState:
        return {"ctx": tools(state["ctx"])}

    def n_responder(state: GraphState) -> GraphState:
        return {"ctx": responder(state["ctx"])}

    def n_judge(state: GraphState) -> GraphState:
        return {"ctx": judge(state["ctx"])}

    # ---- 보조 노드 공통 래퍼: Stage와 동일하게 타이밍/이벤트를 남긴다 ----
    def _timed(name: str, fn: Callable[[PipelineContext], PipelineContext]):
        def node(state: GraphState) -> GraphState:
            ctx = state["ctx"]
            ctx.mark(name)
            ctx.emit(name, "start")
            start = perf_counter()
            ctx = fn(ctx)
            elapsed_ms = round((perf_counter() - start) * 1000, 2)
            timings = ctx.debug.setdefault("node_timings_ms", {})
            timings[name] = round(timings.get(name, 0.0) + elapsed_ms, 2)
            ctx.emit(name, "end", elapsed_ms=elapsed_ms)
            return {"ctx": ctx}
        return node

    # ---- 재시도 준비 노드 (기존 오케스트레이터의 retry 증가 로직과 동일) ----
    def _retry_prep(ctx: PipelineContext) -> PipelineContext:
        ctx.retry_count += 1
        ctx.debug["retry_refined_query"] = ctx.judge.refined_query
        logger.info("judge requested retry (%d/%d)", ctx.retry_count, MAX_JUDGE_RETRIES)
        return ctx

    n_retry_prep = _timed("retry_prep", _retry_prep)

    # ---- 종료 노드 (기존 오케스트레이터의 final_answer 설정과 동일) ----
    def _finalize_blocked(ctx: PipelineContext) -> PipelineContext:
        ctx.final_answer = ctx.security.blocked_reason or "요청을 처리할 수 없습니다."
        return ctx

    def _finalize_memory(ctx: PipelineContext) -> PipelineContext:
        ctx.final_answer = ctx.memory_result.response_text
        return ctx

    def _finalize_answer(ctx: PipelineContext) -> PipelineContext:
        if ctx.judge.decision == JudgeDecision.CLARIFY.value:
            ctx.draft_answer = ctx.judge.clarifying_question or ctx.draft_answer
        ctx.final_answer = ctx.draft_answer
        return ctx

    n_finalize_blocked = _timed("finalize_blocked", _finalize_blocked)
    n_finalize_memory = _timed("finalize_memory", _finalize_memory)
    n_finalize_answer = _timed("finalize_answer", _finalize_answer)

    # ---- 조건부 분기 함수 (읽기 전용) ----
    def route_security(state: GraphState) -> str:
        return "blocked" if not state["ctx"].security.passed else "continue"

    def route_memory(state: GraphState) -> str:
        return "handled" if state["ctx"].memory_result.handled else "continue"

    def route_judge(state: GraphState) -> str:
        ctx = state["ctx"]
        if (
            ctx.judge.decision == JudgeDecision.RETRY.value
            and ctx.retry_count < MAX_JUDGE_RETRIES
        ):
            return "retry"
        return "finalize"

    # ---- 그래프 구성 ----
    graph = StateGraph(GraphState)

    graph.add_node("security", n_security)
    graph.add_node("context_load", n_context_load)
    graph.add_node("planner", n_planner)
    graph.add_node("memory", n_memory)
    graph.add_node("slots", n_slots)
    graph.add_node("tools", n_tools)
    graph.add_node("responder", n_responder)
    graph.add_node("judge", n_judge)
    graph.add_node("retry_prep", n_retry_prep)
    graph.add_node("finalize_blocked", n_finalize_blocked)
    graph.add_node("finalize_memory", n_finalize_memory)
    graph.add_node("finalize_answer", n_finalize_answer)

    graph.set_entry_point("security")

    graph.add_conditional_edges(
        "security",
        route_security,
        {"blocked": "finalize_blocked", "continue": "context_load"},
    )
    graph.add_edge("context_load", "planner")
    graph.add_edge("planner", "memory")
    graph.add_conditional_edges(
        "memory",
        route_memory,
        {"handled": "finalize_memory", "continue": "slots"},
    )
    graph.add_edge("slots", "tools")
    graph.add_edge("tools", "responder")
    graph.add_edge("responder", "judge")
    graph.add_conditional_edges(
        "judge",
        route_judge,
        {"retry": "retry_prep", "finalize": "finalize_answer"},
    )
    graph.add_edge("retry_prep", "tools")

    graph.add_edge("finalize_blocked", END)
    graph.add_edge("finalize_memory", END)
    graph.add_edge("finalize_answer", END)

    return graph.compile()
