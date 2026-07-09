"""RAG 평가기: 하드조건 감점 + 제목/본문 유사 가점 (결정론적 1차 추림).

정책(요구사항 Q10)
- 하드조건은 '메타데이터와 질문슬롯이 둘 다 있을 때만' 적용. 없으면 스킵(평가 제외).
- 불일치 시 점수를 크게 낮춘다. 일치해도 하드조건으로는 가점하지 않는다.
- 제목/본문이 질문과 유사하면(어휘 겹침) 유사도에 가점한다.
- 최종 적합성 판정은 이 후 LLM이 담당(B). 여기서는 빠른 정렬만 한다.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent3.rag.retriever import RagCandidate
from agent3.schemas.pipeline import NormalizedSlots

# 감점 배율
_PENALTY_REGION = 0.3
_PENALTY_AGE = 0.3
_PENALTY_INCOME = 0.3
_PENALTY_FIELD = 0.5
_PENALTY_MARRIAGE = 0.5

# 가점
_BONUS_TITLE_HIT = 0.06
_BONUS_CONTENT_HIT = 0.02
_BONUS_CAP = 0.3


def _to_int(value: Any) -> Optional[int]:
    if value in (None, "", "nan"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _tokenize(text: str) -> list[str]:
    # 한글/영문/숫자 2글자 이상 토큰
    return [t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", text or "")]


def _age_overlaps(slots: NormalizedSlots, meta_min: Optional[int], meta_max: Optional[int]) -> bool:
    q_min = slots.age if slots.age is not None else slots.age_min
    q_max = slots.age if slots.age is not None else slots.age_max
    if q_min is None and q_max is None:
        return True
    lo = meta_min if meta_min is not None else 0
    hi = meta_max if meta_max is not None else 200
    q_lo = q_min if q_min is not None else lo
    q_hi = q_max if q_max is not None else hi
    return not (q_hi < lo or q_lo > hi)


def evaluate_candidate(
    candidate: RagCandidate,
    *,
    query: str,
    slots: NormalizedSlots,
    query_field: Optional[str],
    query_keywords: list[str],
) -> RagCandidate:
    meta = candidate.metadata
    score = candidate.base_sim
    detail: dict[str, Any] = {"base_sim": round(candidate.base_sim, 4), "penalties": [], "bonus": 0.0}

    # ---- 하드조건 감점 (둘 다 있을 때만) ----
    meta_region = str(meta.get("region", "")).strip()
    if slots.region and meta_region and slots.region != meta_region:
        score *= _PENALTY_REGION
        detail["penalties"].append("region")

    meta_age_min = _to_int(meta.get("age_min"))
    meta_age_max = _to_int(meta.get("age_max"))
    if (slots.age is not None or slots.age_min is not None) and (meta_age_min is not None or meta_age_max is not None):
        if not _age_overlaps(slots, meta_age_min, meta_age_max):
            score *= _PENALTY_AGE
            detail["penalties"].append("age")

    meta_income = _to_int(meta.get("income_max"))
    if slots.income is not None and meta_income is not None and slots.income > meta_income:
        score *= _PENALTY_INCOME
        detail["penalties"].append("income")

    meta_field = str(meta.get("field", "")).strip()
    if query_field and meta_field and query_field != meta_field:
        score *= _PENALTY_FIELD
        detail["penalties"].append("field")

    meta_marriage = str(meta.get("marriage", "")).strip()
    if slots.marriage_status and meta_marriage and slots.marriage_status != meta_marriage:
        score *= _PENALTY_MARRIAGE
        detail["penalties"].append("marriage")

    # ---- 유사 가점 (제목/본문 어휘 겹침) ----
    keyword_pool = list(dict.fromkeys([*_tokenize(query), *(query_keywords or [])]))
    title = candidate.title or ""
    content = candidate.content or ""
    bonus = 0.0
    for kw in keyword_pool:
        if kw in title:
            bonus += _BONUS_TITLE_HIT
        elif kw in content:
            bonus += _BONUS_CONTENT_HIT
    bonus = min(bonus, _BONUS_CAP)
    score += bonus
    detail["bonus"] = round(bonus, 4)

    candidate.score = round(score, 4)
    candidate.eval_detail = detail
    return candidate


def evaluate_and_rank(
    candidates: list[RagCandidate],
    *,
    query: str,
    slots: NormalizedSlots,
    query_field: Optional[str],
    query_keywords: list[str],
    top_k: int = 10,
) -> list[RagCandidate]:
    evaluated = [
        evaluate_candidate(
            c,
            query=query,
            slots=slots,
            query_field=query_field,
            query_keywords=query_keywords,
        )
        for c in candidates
    ]
    evaluated.sort(key=lambda c: c.score, reverse=True)
    return evaluated[:top_k]
