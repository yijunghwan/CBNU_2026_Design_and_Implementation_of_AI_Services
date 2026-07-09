"""agent3 서비스: LangGraph 앱 + 메모리 매니저 소유, 턴 처리/영속화/스트리밍 담당.

- 로직은 agent2의 Stage 구현을 그대로 사용하고, 실행 흐름만 LangGraph로 제어한다.
- 요청 1건을 처리하고, 응답 후 세션/장기기억에 저장한다.
- 메모리 명령(장기기억 제어)은 저장에서 제외한다(제어성 대화 오염 방지).
- process_stream: 단계/툴 진행 이벤트를 실시간(SSE)으로 흘려보낸다.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from agent3.schemas.pipeline import PipelineContext
from agent3.store.memory_manager import MemoryManager
from agent3.graph import build_graph


@dataclass
class AgentService:
    manager: MemoryManager = field(default_factory=MemoryManager)

    def __post_init__(self) -> None:
        # LangGraph 컴파일 앱 (기존 Stage 로직을 노드로 사용)
        self.app = build_graph(self.manager)

    def _run_graph(self, ctx: PipelineContext) -> PipelineContext:
        final_state = self.app.invoke({"ctx": ctx})
        return final_state["ctx"]

    # ---- 공통 컨텍스트 ----
    def _make_ctx(self, **kw) -> PipelineContext:
        return PipelineContext(
            user_id=kw["user_id"],
            query=kw["query"],
            user_code=kw.get("user_code"),
            provider=(kw.get("provider") or "openai"),
            model_name=(kw.get("model_name") or "gpt-4o-mini"),
            file_path=kw.get("file_path"),
            file_type=kw.get("file_type"),
        )

    def _persist(self, identity: str, query: str, ctx: PipelineContext) -> None:
        if not ctx.memory_result.handled:
            self.manager.persist_turn(identity, query, ctx.final_answer or "")
        else:
            self.manager.session.append(identity, "user", query)
            self.manager.session.append(identity, "assistant", ctx.final_answer or "")

    def _memory_view(self, identity: str) -> dict[str, Any]:
        state = self.manager.runtime(identity)
        view: dict[str, Any] = {
            "enabled": state.enabled, "profile": {}, "summary": "",
            "recent_count": 0, "recent": [],
        }
        if state.enabled and state.user_code:
            code = state.user_code
            view["profile"] = self.manager.long.get_profile(code)
            view["summary"] = self.manager.long.get_summary(code)
            recent = self.manager.long.get_recent_messages(code, limit=10)
            view["recent_count"] = len(recent)
            view["recent"] = recent[-6:]
        return view

    def _build_result(self, ctx: PipelineContext, identity: str) -> dict[str, Any]:
        memory = self._memory_view(identity)
        memory.update(
            handled=ctx.memory_result.handled,
            action=ctx.memory_result.action,
            status=ctx.memory_result.status,
        )
        return {
            "response": ctx.final_answer,
            "trace": ctx.trace,
            "node_timings_ms": ctx.debug.get("node_timings_ms", {}),
            "plan": {
                "intent": ctx.plan.intent,
                "tools": [t.tool for t in ctx.plan.tool_plan],
                "memory_action": ctx.plan.memory_action,
                "missing_slots": ctx.plan.missing_slots,
                "needs_user_profile": ctx.plan.needs_user_profile,
                "domain_keywords": ctx.plan.domain_keywords,
            },
            "slots": {
                "region": ctx.slots.region,
                "age": ctx.slots.age,
                "income": ctx.slots.income,
                "employment_status": ctx.slots.employment_status,
                "marriage_status": ctx.slots.marriage_status,
                "housing_status": ctx.slots.housing_status,
            },
            "executed_tools": ctx.tools.executed_tools,
            "tool_notes": ctx.tools.notes,
            "rag_evaluation": ctx.tools.rag_evaluation,
            "policies": ctx.tools.rag_policies,
            "contents": ctx.tools.rag_contents,
            "web_results": ctx.tools.web_results,
            "clarify_domain": bool(ctx.debug.get("clarify_domain")),
            "judge": ctx.judge.decision,
            "retry_count": ctx.retry_count,
            "memory": memory,
        }

    # ---- 동기 처리 ----
    def process(self, **kw) -> dict[str, Any]:
        ctx = self._make_ctx(**kw)
        ctx = self._run_graph(ctx)
        identity = self.manager.resolve_identity(kw["user_id"], kw.get("user_code"))
        self._persist(identity, kw["query"], ctx)
        return self._build_result(ctx, identity)

    # ---- 스트리밍 처리 (SSE 이벤트 제너레이터) ----
    def process_stream(self, **kw) -> Iterator[dict[str, Any]]:
        ctx = self._make_ctx(**kw)
        identity = self.manager.resolve_identity(kw["user_id"], kw.get("user_code"))

        events: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue()
        ctx.on_event = lambda ev: events.put({"type": "progress", **ev})
        holder: dict[str, Any] = {}

        def _run() -> None:
            try:
                done = self._run_graph(ctx)
                self._persist(identity, kw["query"], done)
                holder["result"] = self._build_result(done, identity)
            except Exception as e:  # noqa: BLE001
                holder["error"] = str(e)
            finally:
                events.put(None)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        while True:
            ev = events.get()
            if ev is None:
                break
            yield ev

        worker.join(timeout=1)
        if "error" in holder:
            yield {"type": "error", "message": holder["error"]}
        else:
            yield {"type": "final", "result": holder.get("result", {})}
