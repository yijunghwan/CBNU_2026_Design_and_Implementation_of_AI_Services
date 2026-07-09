"""플래너 LLM 검증 (임시)."""

from agent3.pipeline import build_default_pipeline
from agent3.schemas.pipeline import PipelineContext


def run(query: str) -> None:
    pipeline = build_default_pipeline()
    ctx = pipeline.run(PipelineContext(user_id="u-test", query=query))
    plan = ctx.plan
    print("=" * 70)
    print("QUERY:", query)
    print(f"  intent={plan.intent} policy={plan.is_policy_query} profile={plan.needs_user_profile}")
    print(f"  memory_action={plan.memory_action} direct={plan.can_answer_directly}")
    print(f"  tools={[(t.tool, t.query) for t in plan.tool_plan]}")
    print(f"  missing={plan.missing_slots} domain={plan.domain_keywords}")
    print(f"  slots_raw={plan.raw.get('slots')}")
    print(f"  normalized: region={ctx.slots.region} age={ctx.slots.age} field_kw={ctx.slots.domain_keywords[:3]}")
    if ctx.tools.rag_policies:
        print("  TOP:", [f"{x['score']:.2f}:{x['title'][:20]}" for x in ctx.tools.rag_policies[:3]])


if __name__ == "__main__":
    run("서울 사는 27살인데 창업 지원 정책 알려줘")
    run("적금 관련 청년 정책 추천해줘")
    run("안녕 너 뭐 할 수 있어?")
    run("장기기억 켜줘")
    run("청년 월세 지원 최신 공고 있어?")
