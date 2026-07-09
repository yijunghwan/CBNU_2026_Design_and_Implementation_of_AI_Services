"""컨텍스트 윈도우 빌더: 모든 LLM 프롬프트에 공통 주입할 대화 맥락 조립.

재료
- 단기(세션): 최근 N턴 + 단기 요약본
- 장기(ON일 때만): 장기 최근 대화 윈도우 + 장기 요약본 + 장기 프로필

장기기억이 켜져 있으면 최근 대화도 DB에 저장/조회되어 여기 포함된다.
"""

from __future__ import annotations

from typing import Any

from agent3.schemas.pipeline import MemoryContext

SHORT_TURNS = 6
LONG_TURNS = 10
MAX_ITEM_CHARS = 180


def _format_window(history: list[dict[str, str]], limit: int, title: str) -> str:
    if not history:
        return f"{title}: 없음"
    lines = [f"{title}:"]
    for msg in history[-limit:]:
        role = str(msg.get("role", "")).lower()
        role_ko = "사용자" if role == "user" else "어시스턴트" if role == "assistant" else role or "기타"
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > MAX_ITEM_CHARS:
            content = content[:MAX_ITEM_CHARS] + "..."
        lines.append(f"- {role_ko}: {content}")
    return "\n".join(lines) if len(lines) > 1 else f"{title}: 없음"


def _format_profile(profile: dict[str, Any]) -> str:
    if not profile:
        return "장기 프로필: 없음"
    keys = [
        ("age", "나이"), ("region", "지역"), ("income", "월소득"),
        ("employment_status", "고용"), ("marriage_status", "결혼"),
        ("children_count", "자녀수"), ("housing_status", "주거"),
    ]
    parts = [f"{ko}={profile.get(k)}" for k, ko in keys if profile.get(k) not in (None, "", [])]
    return "장기 프로필: " + (", ".join(parts) if parts else "없음")


def build_context_block(
    memory: MemoryContext,
    *,
    short_turns: int = SHORT_TURNS,
    long_turns: int = LONG_TURNS,
    include_profile: bool = True,
) -> str:
    """LLM 프롬프트용 맥락 블록 생성."""
    short_summary = (memory.short_summary or "").strip() or "없음"
    blocks = [
        f"[단기 요약본]\n{short_summary}",
        _format_window(memory.short_window, short_turns, "[최근 단기 대화]"),
    ]

    if memory.long_memory_enabled:
        long_summary = (memory.long_summary or "").strip() or "없음"
        blocks.append(f"[장기 요약본]\n{long_summary}")
        blocks.append(_format_window(memory.long_window, long_turns, "[장기 최근 대화]"))
        if include_profile:
            blocks.append(_format_profile(memory.long_profile))

    return "\n\n".join(blocks)
