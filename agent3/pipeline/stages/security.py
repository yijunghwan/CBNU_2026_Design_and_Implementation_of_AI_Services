"""보안 단계: 민감정보/과부하 규칙 검사 (규칙 기반, LLM 아님).

이 단계는 의도적으로 규칙 기반이다. (사용자 요구: 미들웨어 몇 가지는 키워드 허용)
"""

from __future__ import annotations

import re

from agent3.pipeline.base import Stage
from agent3.schemas.pipeline import PipelineContext, SecurityResult

SENSITIVE_PATTERNS: dict[str, str] = {
    "주민등록번호": r"\d{6}-\d{7}",
    "휴대폰번호": r"01[0-9]-\d{3,4}-\d{4}",
    "여권번호": r"[A-Z]\d{8}",
    "신용카드": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
}

MAX_QUERY_CHARS = 4000


class SecurityStage(Stage):
    name = "security"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        hits: list[str] = []
        for label, pattern in SENSITIVE_PATTERNS.items():
            if re.search(pattern, ctx.query or ""):
                hits.append(label)

        if hits:
            ctx.security = SecurityResult(
                passed=False,
                blocked_reason=(
                    f"입력에 민감정보({', '.join(hits)})가 포함되어 있어요. "
                    "개인 식별정보는 제거한 뒤 다시 질문해주세요."
                ),
                sensitive_hits=hits,
            )
            return ctx

        if len(ctx.query or "") > MAX_QUERY_CHARS:
            ctx.security = SecurityResult(
                passed=False,
                blocked_reason="질문이 너무 길어요. 핵심만 줄여서 다시 보내주세요.",
            )
            return ctx

        ctx.security = SecurityResult(passed=True)
        return ctx
