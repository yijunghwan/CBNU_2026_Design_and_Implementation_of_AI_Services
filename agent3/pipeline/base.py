"""파이프라인 단계(Stage) 공통 계약.

모든 단계는 Stage를 상속하고 run(ctx)에서 PipelineContext를 갱신해 반환한다.
단계는 서로를 직접 호출하지 않는다. (오케스트레이터만 순서를 안다)
__call__에서 실행 시간을 계측해 ctx.debug['node_timings_ms']에 누적한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from agent3.schemas.pipeline import PipelineContext


class Stage(ABC):
    """파이프라인 단일 책임 단계."""

    name: str = "stage"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """컨텍스트를 받아 일부 필드를 채우고 반환한다."""
        raise NotImplementedError

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        ctx.mark(self.name)
        ctx.emit(self.name, "start")
        start = perf_counter()
        result = self.run(ctx)
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        timings = result.debug.setdefault("node_timings_ms", {})
        # 재시도로 같은 노드가 여러 번 실행되면 누적한다.
        timings[self.name] = round(timings.get(self.name, 0.0) + elapsed_ms, 2)
        result.emit(self.name, "end", elapsed_ms=elapsed_ms)
        return result
