"""agent2 수동 스모크 테스트 (임시)."""

from agent3.pipeline import build_default_pipeline
from agent3.schemas.pipeline import PipelineContext


def run(query: str) -> None:
    pipeline = build_default_pipeline()
    ctx = pipeline.run(PipelineContext(user_id="u-test", query=query))
    print("=" * 70)
    print("QUERY:", query)
    print("TRACE:", ctx.trace)
    print("SLOTS:", ctx.slots.region, ctx.slots.age, ctx.slots.domain_keywords)
    print("RAG EVAL:", ctx.tools.rag_evaluation)
    print("TOP POLICIES:")
    for x in ctx.tools.rag_policies[:8]:
        print(
            f"  {x['score']:.3f} | {x['field']:>4} | {x['region']:>2} | "
            f"pen={x['eval']['penalties']} | {x['title'][:40]}"
        )


if __name__ == "__main__":
    run("지금 수원 살고 나이는 20살 적금 관련 정책")
    run("서울 사는 30살인데 창업 지원 정책 알려줘")
