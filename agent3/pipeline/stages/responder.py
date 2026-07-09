"""응답 생성 단계: 사용자가 고른 모델이 툴 결과 + 컨텍스트로 최종 답변 작성.

동작(요구사항 Q14=C + 소프트 안내)
- 개인 맞춤추천인데 핵심 조건이 부족하면: 부족한 조건을 먼저 안내하고 가능한 범위로 답한다.
- 일반 정책검색인데 일부 조건이 없으면: 결과는 보여주되 "정보 주면 더 정확" 소프트 안내.
- 툴이 필요없는 대화(can_answer_directly)면 맥락 기반으로 간결히 답한다.
"""

from __future__ import annotations

import logging

from agent3.llm import build_context_block, create_llm, invoke_text
from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_MAX_POLICY_REFS = 6
_MAX_CONTENT_REFS = 4
_MAX_WEB_REFS = 3


def _format_policies(policies: list[dict]) -> str:
    if not policies:
        return "정책 검색 결과: 없음"
    lines = ["정책 검색 결과:"]
    for i, p in enumerate(policies[:_MAX_POLICY_REFS], start=1):
        lines.append(
            f"[{i}] {p.get('title','')} | 분야:{p.get('field','') or '-'} | "
            f"지역:{p.get('region','') or '-'} | 내용:{p.get('content','')[:160]}"
        )
    return "\n".join(lines)


def _format_contents(contents: list[dict]) -> str:
    if not contents:
        return ""
    lines = ["컨텐츠(공지/뉴스) 검색 결과:"]
    for i, c in enumerate(contents[:_MAX_CONTENT_REFS], start=1):
        lines.append(f"[{i}] {c.get('title','')} | 내용:{c.get('content','')[:140]}")
    return "\n".join(lines)


def _format_web(web: list[dict]) -> str:
    if not web:
        return ""
    lines = ["웹 검색 결과:"]
    for i, w in enumerate(web[:_MAX_WEB_REFS], start=1):
        lines.append(f"[{i}] {w.get('title','')} | {w.get('content','')[:140]}")
    return "\n".join(lines)


_PROMPT = """당신은 '청년정책 추천 에이전트'입니다. 아래 자료를 근거로 사용자 질문에 최종 답변을 작성하세요.

원칙:
- 사용자 질문이 최우선입니다. 근거 자료에 없는 내용은 지어내지 마세요.
- 개인 맞춤추천({needs_profile})이고 핵심 조건이 부족하면, 먼저 부족한 조건을 짧게 안내한 뒤 가능한 범위로 답하세요.
- 일반 정책검색이라도 조건이 부족하면 결과를 보여준 뒤 "지역/나이/소득 등을 알려주면 더 정확해요"라고 한 줄로 덧붙이세요.
- 부족 조건(있으면): {missing}
- 도구 결과를 그대로 나열하지 말고, 정리된 추천/설명으로 답하세요.
- 정책이 있으면 아래 형식을 지키세요:
[추천 요약] 한두 문장
[추천 정책]
1. 정책명 - 왜 맞는지 한 문장
2. 정책명 - 왜 맞는지 한 문장
3. 정책명 - 왜 맞는지 한 문장
[추가로 알면 좋은 정보] 1~2줄 (또는 '현재 정보로도 기본 추천 가능')
- 전체 8줄 안팎으로 간결하게.

대화 맥락:
{context}

사용자 질문: {query}

{policy_block}
{content_block}
{web_block}
{file_block}"""

_DIRECT_PROMPT = """당신은 청년정책 추천 에이전트입니다. 아래 맥락을 참고해 사용자 질문에 간결히 답하세요.
- 정책과 무관한 잡담/기능질문이면 짧게 답하고, 필요하면 정책 질문 예시 1개만 제시하세요.
- 과장 없이 사실 위주로 답하세요.

대화 맥락:
{context}

사용자 질문: {query}"""


class ResponderStage(Stage):
    name = "responder"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        context_block = build_context_block(ctx.memory)
        llm = create_llm(ctx.provider, ctx.model_name, temperature=0.3, max_tokens=1000)

        # Q18(A): 분야가 없는 맞춤추천이면 관심분야를 되묻는다(설명 충분히)
        if ctx.debug.get("clarify_domain"):
            ctx.draft_answer = (
                "맞춤 정책을 추천해드리려면 관심 분야를 먼저 알려주세요.\n"
                "- 선택지: 주거 / 취업 / 금융 / 교육 / 복지문화 / 참여권리\n"
                "예를 들어 '취업 지원 정책 알려줘'처럼 말씀해 주시면 바로 찾아드려요.\n"
                "\n"
                "추가로 나이, 결혼여부, 소득, 주거상태 같은 정보를 함께 알려주시면 "
                "조건에 더 잘 맞는 정책만 골라 정확도가 높아집니다."
            )
            return ctx

        has_results = bool(ctx.tools.rag_policies or ctx.tools.rag_contents or ctx.tools.web_results)
        file_block = ""
        if ctx.tools.file_text:
            file_block = f"업로드 파일 내용(발췌):\n{ctx.tools.file_text[:1500]}"

        if ctx.plan.can_answer_directly and not has_results and not file_block:
            prompt = _DIRECT_PROMPT.format(context=context_block, query=ctx.query)
            ctx.draft_answer = invoke_text(llm, prompt) or "무엇을 도와드릴까요?"
            return ctx

        prompt = _PROMPT.format(
            needs_profile="예" if ctx.plan.needs_user_profile else "아니오",
            missing=", ".join(ctx.plan.missing_slots) if ctx.plan.missing_slots else "없음",
            context=context_block,
            query=ctx.query,
            policy_block=_format_policies(ctx.tools.rag_policies),
            content_block=_format_contents(ctx.tools.rag_contents),
            web_block=_format_web(ctx.tools.web_results),
            file_block=file_block,
        )
        answer = invoke_text(llm, prompt)
        ctx.draft_answer = answer or "관련 정보를 정리하지 못했어요. 조건을 조금 더 알려주시겠어요?"
        return ctx
