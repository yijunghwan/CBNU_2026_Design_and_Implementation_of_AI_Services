"""RAG 검색기: 정책/컨텐츠 벡터스토어에서 후보를 뽑는다.

- 벡터 유사도 검색 후 doc_id 기준으로 문서 단위 dedup(최고점 청크 유지).
- 반환은 RagCandidate 리스트 (메타데이터 + 기본 유사도 점수).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from agent3.config import settings


@dataclass
class RagCandidate:
    doc_id: str
    title: str
    content: str
    url: str
    base_sim: float                       # 0~1 (검색 유사도)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0                    # 평가 후 최종 점수
    eval_detail: dict[str, Any] = field(default_factory=dict)


def _distance_to_sim(distance: float) -> float:
    # Chroma cosine distance(작을수록 유사) → 0~1 유사도
    try:
        return 1.0 / (1.0 + float(distance))
    except (TypeError, ValueError):
        return 0.0


class RagRetriever:
    """단일 컬렉션 검색기 (정책 또는 컨텐츠)."""

    def __init__(self, persist_dir: Path):
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self._store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=self._embeddings,
        )

    def search(self, query: str, k: int = 20, fetch_k: int = 60) -> list[RagCandidate]:
        raw = self._store.similarity_search_with_score(query, k=fetch_k)

        best_by_doc: dict[str, RagCandidate] = {}
        for doc, distance in raw:
            meta = dict(doc.metadata or {})
            doc_id = str(meta.get("doc_id") or id(doc))
            sim = _distance_to_sim(distance)

            existing = best_by_doc.get(doc_id)
            if existing and existing.base_sim >= sim:
                continue

            best_by_doc[doc_id] = RagCandidate(
                doc_id=doc_id,
                title=str(meta.get("title", "")),
                content=doc.page_content,
                url=str(meta.get("url", "")),
                base_sim=sim,
                metadata=meta,
            )

        candidates = sorted(best_by_doc.values(), key=lambda c: c.base_sim, reverse=True)
        return candidates[:k]


_POLICY_RETRIEVER: Optional[RagRetriever] = None
_CONTENT_RETRIEVER: Optional[RagRetriever] = None


def get_policy_retriever() -> RagRetriever:
    global _POLICY_RETRIEVER
    if _POLICY_RETRIEVER is None:
        _POLICY_RETRIEVER = RagRetriever(settings.policy_store_dir)
    return _POLICY_RETRIEVER


def get_content_retriever() -> RagRetriever:
    global _CONTENT_RETRIEVER
    if _CONTENT_RETRIEVER is None:
        _CONTENT_RETRIEVER = RagRetriever(settings.content_store_dir)
    return _CONTENT_RETRIEVER
