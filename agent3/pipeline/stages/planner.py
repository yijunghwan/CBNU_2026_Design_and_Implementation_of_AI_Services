"""플래너 단계: 작은 LLM이 의도/메모리명령/툴계획/슬롯을 자율 결정.

핵심: 키워드 하드코딩 대신 LLM이 문장 전체 의도를 보고 계획한다.
산출(JSON 한 줄):
{
  "intent": "policy|content|term|latest|smalltalk|capability|general",
  "is_policy_query": bool,
  "needs_user_profile": bool,
  "memory_action": "none|enable|disable|show_profile|update_profile|delete_profile|delete_history|delete_all",
  "can_answer_directly": bool,
  "tool_plan": [{"tool":"rag_policy_search|rag_content_search|web_search","reason":"...","query":"..."}],
  "missing_slots": ["지역","나이",...],
  "domain_keywords": ["금융","저축",...],
  "slots": {"region":..,"age":..,"income":..,"employment_status":..,"marriage_status":..,"children_count":..,"housing_status":..}
}
"""

from __future__ import annotations

import logging

from agent3.config import settings
from agent3.llm import build_context_block, create_llm, invoke_json
from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import (
    MemoryAction,
    PipelineContext,
    PlannerDecision,
    ToolStep,
)

logger = logging.getLogger(__name__)

_VALID_TOOLS = {"rag_policy_search", "rag_content_search", "web_search"}
_VALID_MEMORY = {a.value for a in MemoryAction}

_PROMPT = """당신은 청년정책 에이전트의 '계획기'입니다. 사용자의 현재 질문 의도를 분석해 실행 계획을 JSON 한 줄로만 반환하세요.

반드시 아래 스키마의 JSON 객체 하나만 출력하세요:
{{"intent":"policy|content|term|latest|smalltalk|capability|general","is_policy_query":bool,"needs_user_profile":bool,"memory_action":"none|enable|disable|show_profile|update_profile|delete_profile|delete_history|delete_all","can_answer_directly":bool,"tool_plan":[{{"tool":"rag_policy_search|rag_content_search|web_search","reason":"짧게","query":"검색어(선택)"}}],"missing_slots":["부족한 조건"],"domain_keywords":["의미확장 키워드"],"slots":{{"region":null,"age":null,"income":null,"employment_status":null,"marriage_status":null,"children_count":null,"housing_status":null}}}}

판단 규칙:
- 정책 추천/검색이면 tool_plan에 rag_policy_search. 개인 맞춤(내 상황 기반)이면 needs_user_profile=true.
- 공지/모집/후기/사례/행사 같은 컨텐츠성이면 rag_content_search.
- 최신/최근/마감/뉴스/시사 또는 정책용어 보완이 필요하면 web_search를 추가.
- 단순 인사/잡담/기능질문/자기소개처럼 툴이 필요없으면 can_answer_directly=true, tool_plan=[].
- 개인정보 저장/수정/삭제, 장기기억 켜기/끄기/조회 의도가 명확할 때만 memory_action 지정(아니면 none).
- 맞춤추천인데 조건이 부족하면 missing_slots에 부족 항목(예: 지역, 나이, 소득)을 넣으세요.
- domain_keywords: 검색 정확도를 위한 의미확장(예: 적금→금융,저축,자산형성). 없으면 [].
- slots: 현재 질문에 '명시된' 값만 추출(추측 금지). 없으면 null. 나이는 정수만.

대화 맥락:
{context}

사용자 질문: {query}"""


def _default_decision() -> dict:
    return {
        "intent": "general",
        "is_policy_query": False,
        "needs_user_profile": False,
        "memory_action": "none",
        "can_answer_directly": True,
        "tool_plan": [],
        "missing_slots": [],
        "domain_keywords": [],
        "slots": {},
    }


class PlannerStage(Stage):
    name = "planner"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        context_block = build_context_block(ctx.memory)
        prompt = _PROMPT.format(context=context_block, query=ctx.query)

        llm = create_llm(
            settings.default_provider,
            settings.default_model,
            temperature=0,
            max_tokens=500,
        )
        raw = invoke_json(llm, prompt, _default_decision())
        ctx.plan = self._to_decision(raw)
        ctx.debug["planner_raw"] = raw
        return ctx

    @staticmethod
    def _to_decision(raw: dict) -> PlannerDecision:
        memory_action = str(raw.get("memory_action", "none")).strip()
        if memory_action not in _VALID_MEMORY:
            memory_action = MemoryAction.NONE.value

        tool_plan: list[ToolStep] = []
        for item in raw.get("tool_plan", []) or []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            if tool not in _VALID_TOOLS:
                continue
            tool_plan.append(
                ToolStep(
                    tool=tool,
                    reason=str(item.get("reason", "")).strip(),
                    query=(str(item.get("query")).strip() or None) if item.get("query") else None,
                )
            )

        domain_keywords = [str(k).strip() for k in (raw.get("domain_keywords") or []) if str(k).strip()]
        missing_slots = [str(s).strip() for s in (raw.get("missing_slots") or []) if str(s).strip()]
        slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else {}

        return PlannerDecision(
            intent=str(raw.get("intent", "general")).strip() or "general",
            is_policy_query=bool(raw.get("is_policy_query", False)),
            needs_user_profile=bool(raw.get("needs_user_profile", False)),
            memory_action=memory_action,
            can_answer_directly=bool(raw.get("can_answer_directly", not tool_plan)),
            tool_plan=tool_plan,
            missing_slots=missing_slots,
            domain_keywords=domain_keywords,
            raw={"slots": slots, "planner": raw},
        )
