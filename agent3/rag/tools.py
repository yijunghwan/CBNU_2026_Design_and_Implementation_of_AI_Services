"""RAG 툴: 정책/컨텐츠 검색 + 결정론적 평가(1차 추림).

최종 적합성 LLM 판정은 상위 단계(판정 LLM)에서 수행한다.
여기서는 검색 → 하드조건 감점/유사 가점 → 상위 후보 반환까지 담당한다.
"""

from __future__ import annotations

from typing import Any, Optional

from agent3.mapping import infer_query_field
from agent3.rag.evaluator import evaluate_and_rank
from agent3.rag.retriever import RagCandidate, get_content_retriever, get_policy_retriever
from agent3.schemas.pipeline import NormalizedSlots


def _candidate_to_dict(c: RagCandidate) -> dict[str, Any]:
    meta = c.metadata
    return {
        "doc_id": c.doc_id,
        "title": c.title,
        "content": c.content[:500],
        "url": c.url,
        "field": meta.get("field", ""),
        "region": meta.get("region", ""),
        "category": meta.get("category_raw", ""),
        "score": c.score,
        "base_sim": c.base_sim,
        "eval": c.eval_detail,
    }


def policy_rag_search(
    query: str,
    *,
    slots: NormalizedSlots,
    query_keywords: Optional[list[str]] = None,
    k: int = 20,
    top_k: int = 10,
) -> dict[str, Any]:
    keywords = list(query_keywords or slots.domain_keywords or [])
    query_field = infer_query_field(keywords, query)

    retriever = get_policy_retriever()
    candidates = retriever.search(query, k=k)
    ranked = evaluate_and_rank(
        candidates,
        query=query,
        slots=slots,
        query_field=query_field,
        query_keywords=keywords,
        top_k=top_k,
    )
    return {
        "status": "success",
        "count": len(ranked),
        "query_field": query_field,
        "policies": [_candidate_to_dict(c) for c in ranked],
    }


def content_rag_search(
    query: str,
    *,
    query_keywords: Optional[list[str]] = None,
    k: int = 15,
    top_k: int = 6,
) -> dict[str, Any]:
    keywords = list(query_keywords or [])
    retriever = get_content_retriever()
    candidates = retriever.search(query, k=k)
    # 컨텐츠는 하드조건이 없으므로 빈 슬롯으로 유사 가점만 적용
    ranked = evaluate_and_rank(
        candidates,
        query=query,
        slots=NormalizedSlots(),
        query_field=None,
        query_keywords=keywords,
        top_k=top_k,
    )
    return {
        "status": "success",
        "count": len(ranked),
        "contents": [_candidate_to_dict(c) for c in ranked],
    }
