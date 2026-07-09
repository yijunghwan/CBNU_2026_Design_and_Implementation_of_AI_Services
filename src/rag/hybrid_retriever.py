"""
RAG 하이브리드 리트리버 - 벡터 + BM25 검색
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters.base import BM25Retriever
except ImportError:
    try:
        from langchain.retrievers import BM25Retriever
    except ImportError:
        from langchain_community.retrievers import BM25Retriever
from src.rag.vectorstore import RAGVectorStore


@dataclass
class SearchResult:
    """검색 결과"""
    document: Document
    score: float  # 0.0 ~ 1.0
    source: str   # "vector", "bm25"


class HybridRetriever:
    """
    하이브리드 리트리버
    
    - 벡터 검색 + BM25 키워드 검색
    - 가중치 기반 병합
    - 메타데이터 필터링
    - 재랭킹 (선택적)
    """

    def __init__(
        self,
        vectorstore: RAGVectorStore,
        alpha: float = 0.5,  # 벡터 검색 가중치
    ):
        """
        Args:
            vectorstore: RAGVectorStore 인스턴스
            alpha: 벡터 검색 가중치 (1-alpha는 BM25 가중치)
        """
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.bm25_retriever: Optional[BM25Retriever] = None

    def initialize_bm25(self, documents: List[Document]):
        """
        BM25 리트리버 초기화
        
        Args:
            documents: 문서 리스트
        """
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        print(f"✅ BM25 리트리버 초기화: {len(documents)} 문서")

    def _normalize_scores(
        self,
        scores: List[float]
    ) -> List[float]:
        """
        점수 정규화 (0.0 ~ 1.0)
        
        Args:
            scores: 원본 점수 리스트
        
        Returns:
            정규화된 점수
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            return [0.5] * len(scores)
        
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def search(
        self,
        query: str,
        k: int = 5,
        use_vector: bool = True,
        use_bm25: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        하이브리드 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            use_vector: 벡터 검색 사용 여부
            use_bm25: BM25 검색 사용 여부
            filter_dict: 메타데이터 필터 (현재 미지원)
        
        Returns:
            SearchResult 리스트 (점수 기준 정렬)
        """
        results: Dict[str, Tuple[Document, float, str]] = {}  # id -> (doc, score, source)
        
        # 벡터 검색
        if use_vector:
            try:
                vector_results = self.vectorstore.search(query, k=k*2)  # 여유 있게 조회
                
                vector_scores = [score for _, score in vector_results]
                normalized_vector_scores = self._normalize_scores(vector_scores)
                
                for (doc, _), norm_score in zip(vector_results, normalized_vector_scores):
                    doc_id = doc.metadata.get("source_doc_id", id(doc))
                    weighted_score = norm_score * self.alpha
                    
                    results[doc_id] = (doc, weighted_score, "vector")
            
            except Exception as e:
                print(f"⚠️ 벡터 검색 실패: {e}")
        
        # BM25 검색
        if use_bm25 and self.bm25_retriever:
            try:
                if hasattr(self.bm25_retriever, "get_relevant_documents"):
                    bm25_results = self.bm25_retriever.get_relevant_documents(query)[:k*2]
                else:
                    bm25_results = self.bm25_retriever.invoke(query)[:k*2]
                
                # BM25는 스코어를 반환하지 않으므로 순서 기반 점수 계산
                bm25_scores = [1.0 - (i / len(bm25_results)) for i in range(len(bm25_results))]
                
                for doc, bm25_score in zip(bm25_results, bm25_scores):
                    doc_id = doc.metadata.get("source_doc_id", id(doc))
                    weighted_score = bm25_score * (1 - self.alpha)
                    
                    if doc_id in results:
                        # 이미 벡터 검색에서 나온 문서면 점수 합산
                        doc, vector_score, _ = results[doc_id]
                        results[doc_id] = (doc, vector_score + weighted_score, "hybrid")
                    else:
                        results[doc_id] = (doc, weighted_score, "bm25")
            
            except Exception as e:
                print(f"⚠️ BM25 검색 실패: {e}")
        
        # 점수 기준 정렬 및 상위 k개 선택
        sorted_results = sorted(
            results.values(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # SearchResult 객체로 변환
        return [
            SearchResult(
                document=doc,
                score=min(1.0, score),  # 최대 1.0으로 정규화
                source=source
            )
            for doc, score, source in sorted_results
        ]

    def search_with_filters(
        self,
        query: str,
        k: int = 5,
        region_filter: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        income_max: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        필터를 포함한 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            region_filter: 지역 필터
            age_min: 최소 나이
            age_max: 최대 나이
            income_max: 최대 소득
        
        Returns:
            필터링된 SearchResult 리스트
        """
        from src.utils.region_mapper import normalize_region
        canonical_region_filter = normalize_region(region_filter) if region_filter else None
        # 먼저 하이브리드 검색
        results = self.search(query, k=k*3, use_vector=True, use_bm25=True)
        
        # 메타데이터 필터링
        filtered_results = []
        for result in results:
            metadata = result.document.metadata
            
            # 지역 필터
            if canonical_region_filter:
                doc_region = normalize_region(str(metadata.get("region", "") or ""))
                # 지역 메타데이터가 없는 문서는 필터 제외하지 않고 유지한다.
                if doc_region and doc_region != canonical_region_filter:
                    continue
            
            # 나이 필터 (eligibility_age)
            if age_min is not None or age_max is not None:
                try:
                    age_lo = metadata.get("eligibility_age_min")
                    age_hi = metadata.get("eligibility_age_max")
                    if age_lo is not None and age_max is not None and age_lo > age_max:
                        continue
                    if age_hi is not None and age_min is not None and age_hi < age_min:
                        continue
                except:
                    pass
            
            # 소득 필터
            if income_max is not None:
                try:
                    eligibility_income = metadata.get("eligibility_income_max")
                    if eligibility_income is not None and income_max > eligibility_income:
                        continue
                except:
                    pass
            
            filtered_results.append(result)
        
        return filtered_results[:k]


class RRFRetriever:
    """
    Reciprocal Rank Fusion (RRF) 기반 리트리버
    
    여러 검색 결과를 통계적으로 병합
    """

    def __init__(self, k: int = 60):  # RRF 상수
        """
        Args:
            k: RRF 상수 (일반적으로 60)
        """
        self.k = k
        self.retrievers: List[Any] = []

    def add_retriever(self, retriever):
        """리트리버 추가"""
        self.retrievers.append(retriever)

    def rrf_score(self, rank: int) -> float:
        """
        RRF 점수 계산
        
        Args:
            rank: 순위 (0-indexed)
        
        Returns:
            RRF 점수 (0.0 ~ 1.0)
        """
        return 1.0 / (self.k + rank + 1)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        RRF 병합 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
        
        Returns:
            SearchResult 리스트
        """
        fused_scores: Dict[str, Tuple[Document, float]] = {}
        
        for retriever in self.retrievers:
            try:
                results = retriever.search(query, k=k*2)
                
                for rank, result in enumerate(results):
                    doc_id = result.document.metadata.get("source_doc_id", id(result.document))
                    rrf_score = self.rrf_score(rank)
                    
                    if doc_id in fused_scores:
                        doc, existing_score = fused_scores[doc_id]
                        fused_scores[doc_id] = (doc, existing_score + rrf_score)
                    else:
                        fused_scores[doc_id] = (result.document, rrf_score)
            
            except Exception as e:
                print(f"⚠️ 리트리버 검색 실패: {e}")
        
        # 점수 기준 정렬
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [
            SearchResult(
                document=doc,
                score=min(1.0, score),
                source="rrf"
            )
            for doc, score in sorted_results
        ]


def create_hybrid_retriever(
    vectorstore: RAGVectorStore,
    documents: List[Document],
    alpha: float = 0.5
) -> HybridRetriever:
    """하이브리드 리트리버 팩토리"""
    retriever = HybridRetriever(vectorstore, alpha=alpha)
    retriever.initialize_bm25(documents)
    return retriever
