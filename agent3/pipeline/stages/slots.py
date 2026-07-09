"""슬롯 정규화 단계: 맵핑 테이블로 표준값만 확정 (결정론적).

우선순위: 현재 질문에서 추출한 값 > 장기 프로필(개인화 필요할 때만).
나이는 age_min/age_max 구조화 필터로만 사용 → 검색어 텍스트 오염 방지.
"""

from __future__ import annotations

from agent3.mapping import SlotNormalizer
from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import PipelineContext


class SlotStage(Stage):
    name = "slots"

    def __init__(self) -> None:
        self.normalizer = SlotNormalizer()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # 1) 현재 질문에서 추출(결정론적)
        query_slots = self.normalizer.from_text(ctx.query)

        # 2) 플래너가 뽑은 원시 슬롯이 있으면 병합
        planner_raw = ctx.plan.raw.get("slots") if isinstance(ctx.plan.raw, dict) else None
        if isinstance(planner_raw, dict):
            planner_slots = self.normalizer.from_raw(planner_raw)
            query_slots = self.normalizer.merge(query_slots, planner_slots)

        # 3) 개인화가 필요하고 장기 프로필이 있으면 보조로 병합(질문 값이 우선)
        if ctx.plan.needs_user_profile and ctx.memory.long_profile:
            profile_slots = self.normalizer.from_raw(ctx.memory.long_profile)
            query_slots = self.normalizer.merge(profile_slots, query_slots)

        # 4) 플래너 도메인 키워드 확장 병합
        if ctx.plan.domain_keywords:
            query_slots.domain_keywords = list(
                dict.fromkeys([*query_slots.domain_keywords, *ctx.plan.domain_keywords])
            )

        ctx.slots = query_slots
        return ctx
