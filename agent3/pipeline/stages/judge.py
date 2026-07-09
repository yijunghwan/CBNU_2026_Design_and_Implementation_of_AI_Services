"""판정 단계: 별도 LLM이 질문-근거-답변 정합성을 평가하고 루프를 제한.

결정
- accept: 답변 채택
- retry: 검색어 재해석 후 1회 재탐색 (refined_query)
- clarify: 질문이 모호하면 되물음 (clarifying_question)

루프 상한은 오케스트레이터가 강제한다(무한 루프 방지).
"""

from __future__ import annotations

import logging

from agent3.config import settings
from agent3.llm import build_context_block, create_llm, invoke_json
from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import JudgeDecision, JudgeResult, PipelineContext

logger = logging.getLogger(__name__)

_MAX_RETRY = 1

_PROMPT = """당신은 최종 응답 품질 평가자입니다. 현재 답변이 질문/대화맥락과 맞는지 판단하세요.
JSON 한 줄만 반환:
{{"decision":"accept|retry|clarify","reason":"짧게","refined_query":"재검색어(있으면)","clarifying_question":"되물음(있으면)"}}

규칙:
- 답변이 질문 핵심을 놓쳤거나 엉뚱하면 retry (refined_query에 현재 질문 재해석을 구체적으로).
- 질문 자체가 너무 모호해 답을 못 정하면 clarify (clarifying_question 한 문장).
- 질문에 나이/지역/분야 등 실행 가능한 조건이 이미 있으면 clarify보다 accept/retry 우선.
- 충분히 맞으면 accept.
- 현재 재시도 {retry}/{max_retry}회. 상한 도달 시 retry 대신 accept 또는 clarify.

대화 맥락:
{context}

사용자 질문: {query}
실행 툴: {tools}
답변 초안:
{draft}"""


class JudgeStage(Stage):
    name = "judge"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # 메모리/직접응답/분야되물음은 결정론적이라 평가 생략
        if ctx.debug.get("clarify_domain"):
            ctx.judge = JudgeResult(decision=JudgeDecision.ACCEPT.value, reason="clarify-domain")
            return ctx
        if ctx.plan.can_answer_directly and not ctx.tools.executed_tools:
            ctx.judge = JudgeResult(decision=JudgeDecision.ACCEPT.value, reason="direct-answer")
            return ctx

        context_block = build_context_block(ctx.memory)
        prompt = _PROMPT.format(
            retry=ctx.retry_count,
            max_retry=_MAX_RETRY,
            context=context_block,
            query=ctx.query,
            tools=ctx.tools.executed_tools,
            draft=(ctx.draft_answer or "")[:1500],
        )
        llm = create_llm(settings.default_provider, settings.default_model, temperature=0, max_tokens=300)
        raw = invoke_json(llm, prompt, {"decision": "accept", "reason": "fallback"})

        decision = str(raw.get("decision", "accept")).strip().lower()
        if decision not in {"accept", "retry", "clarify"}:
            decision = "accept"
        # 상한 초과 retry는 accept로 강등
        if decision == "retry" and ctx.retry_count >= _MAX_RETRY:
            decision = "accept"

        ctx.judge = JudgeResult(
            decision=decision,
            reason=str(raw.get("reason", "")).strip(),
            refined_query=(str(raw.get("refined_query", "")).strip() or None),
            clarifying_question=(str(raw.get("clarifying_question", "")).strip() or None),
        )
        return ctx
