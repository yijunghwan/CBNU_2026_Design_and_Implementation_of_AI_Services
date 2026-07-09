"""툴 실행 단계: 플래너의 tool_plan을 실행하고 결과를 모은다.

구현된 툴
- rag_policy_search: 정책 전용 RAG (하드조건 감점 + 유사 가점 평가 포함)
- rag_content_search: 컨텐츠(공지/뉴스) 전용 RAG (유사 가점만)

RAG 평가 정책(요구사항 Q10)
- 나이/결혼/지역/소득/분야 하드조건 불일치 → 점수 크게 감점 (메타/슬롯 없으면 스킵)
- 제목/본문 의미 유사 → 가점
- 최종 적합성은 이후 판정 LLM이 결정 (여기선 1차 추림)

TODO(다음 단계): web_search / file_parse 실제 연결.
"""

from __future__ import annotations

import logging

from agent3.mapping import infer_query_field
from agent3.pipeline.base import Stage
from agent3.rag.tools import content_rag_search, policy_rag_search
from agent3.schemas.pipeline import PipelineContext, ToolName, ToolResults
from agent3.tools.file_parse import parse_file
from agent3.tools.web_search import web_search

logger = logging.getLogger(__name__)


class ToolStage(Stage):
    name = "tools"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        results = ToolResults()

        # 업로드된 파일이 있으면 항상 먼저 파싱해 응답 근거로 활용
        if ctx.file_path:
            ctx.emit("tool", "start", tool="file_parse")
            parsed = parse_file(ctx.file_path, ctx.file_type)
            results.file_text = parsed.get("text") or None
            results.executed_tools.append(ToolName.FILE_PARSE.value)
            results.notes.append(f"file:{parsed.get('status')}")
            ctx.emit("tool", "end", tool="file_parse", status=parsed.get("status"))

        # 재시도 시 판정 LLM이 다듬은 쿼리를 우선 사용
        search_query = str(ctx.debug.get("retry_refined_query") or ctx.query)
        keywords = list(ctx.slots.domain_keywords or ctx.plan.domain_keywords or [])
        query_field = infer_query_field(keywords, search_query)

        # Q18(A): 개인 맞춤추천인데 분야가 전혀 없으면 되물음(clarify)
        if ctx.plan.needs_user_profile and not query_field and not keywords:
            ctx.debug["clarify_domain"] = True
            ctx.tools = results
            return ctx

        if ctx.plan.can_answer_directly and not ctx.plan.tool_plan:
            ctx.tools = results
            return ctx

        for step in ctx.plan.tool_plan:
            tool = step.tool
            ctx.emit("tool", "start", tool=tool)
            try:
                if tool == ToolName.RAG_POLICY_SEARCH.value:
                    res = policy_rag_search(
                        step.query or search_query,
                        slots=ctx.slots,
                        query_keywords=keywords,
                        top_k=20,
                    )
                    results.rag_policies = res.get("policies", [])
                    results.rag_evaluation["policy_query_field"] = res.get("query_field")
                    results.executed_tools.append(tool)

                elif tool == ToolName.RAG_CONTENT_SEARCH.value:
                    res = content_rag_search(
                        step.query or search_query,
                        query_keywords=keywords,
                    )
                    results.rag_contents = res.get("contents", [])
                    results.executed_tools.append(tool)

                elif tool == ToolName.WEB_SEARCH.value:
                    res = web_search(step.query or search_query)
                    results.web_results = res.get("results", [])
                    results.notes.append(f"web:{res.get('status')}")
                    results.executed_tools.append(tool)

                ctx.emit("tool", "end", tool=tool, status="success")
            except Exception as e:  # noqa: BLE001
                logger.warning("tool %s failed: %s", tool, e)
                results.notes.append(f"error:{tool}:{e}")
                ctx.emit("tool", "end", tool=tool, status="error")

        ctx.tools = results
        return ctx
