"""전체 파이프라인(플래너→툴→응답→판정) 최종 답변 검증 (임시)."""

from agent3.pipeline import build_default_pipeline
from agent3.schemas.pipeline import PipelineContext


def run(query: str) -> None:
    pipeline = build_default_pipeline()
    ctx = pipeline.run(PipelineContext(user_id="u-test", query=query))
    print("=" * 72)
    print("QUERY:", query)
    print(f"plan: intent={ctx.plan.intent} tools={[t.tool for t in ctx.plan.tool_plan]} "
          f"missing={ctx.plan.missing_slots} judge={ctx.judge.decision}")
    print("-" * 72)
    print(ctx.final_answer)


if __name__ == "__main__":
    run("서울 사는 27살인데 창업 지원 정책 알려줘")
    run("적금 관련 청년 정책 추천해줘")
    run("나한테 맞는 정책 추천해줘")
    run("안녕 너 뭐 할 수 있어?")
