"""
정책 검색 도구 - RAG 파이프라인을 이용한 정책 검색
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import base64
import binascii
from pathlib import Path
import re

import pandas as pd

from src.utils.region_mapper import normalize_region


@dataclass
class PolicySearchInput:
    """정책 검색 입력"""
    query: str  # 검색 쿼리
    region: Optional[str] = None  # 지역 필터
    age_min: Optional[int] = None  # 최소 나이
    age_max: Optional[int] = None  # 최대 나이
    income_max: Optional[float] = None  # 최대 소득
    emphasis_terms: Optional[List[str]] = None  # 강조할 조건
    k: int = 5  # 반환할 정책 수


@dataclass
class PolicySearchResult:
    """정책 검색 결과"""
    policy_id: str
    title: str
    content: str
    institution: str
    region: str
    url: str
    score: float
    category: Optional[str] = None
    source: Optional[str] = None
    image_url: Optional[str] = None


class PolicySearchTool:
    """
    정책 검색 도구
    
    RAG 파이프라인을 사용하여 사용자 질문에 맞는 정책을 검색
    - 벡터 + BM25 하이브리드 검색
    - Cross-encoder 재랭킹
    - 지역, 나이, 소득 필터링
    """
    
    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: RAGPipeline 인스턴스
        """
        self.rag_pipeline = rag_pipeline
        self._content_attachment_df = None
        self._content_image_url_cache: Dict[str, str] = {}
        self._content_image_dir = Path("src/static/content_images")

    def _get_content_attachment_df(self):
        if self._content_attachment_df is not None:
            return self._content_attachment_df

        csv_path = Path("data/raw/youth_content.csv")
        if not csv_path.exists():
            self._content_attachment_df = pd.DataFrame(columns=["pstSn", "atchFile"])
            return self._content_attachment_df

        try:
            df = pd.read_csv(csv_path, usecols=["pstSn", "atchFile"], dtype=str, encoding="utf-8-sig")
            self._content_attachment_df = df.fillna("")
        except Exception:
            self._content_attachment_df = pd.DataFrame(columns=["pstSn", "atchFile"])

        return self._content_attachment_df

    def _mime_to_ext(self, mime: str) -> str:
        if mime == "image/jpeg":
            return "jpg"
        if mime == "image/svg+xml":
            return "svg"
        if mime.startswith("image/"):
            return mime.split("/", 1)[1]
        return "png"

    def _normalize_url(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    def _extract_region_token(self, query: str) -> str:
        if not query:
            return ""
        return normalize_region(query) or ""

    def _score_emphasis_match(self, result: PolicySearchResult, emphasis_terms: List[str]) -> int:
        haystack = f"{result.title} {result.content} {result.category or ''}"
        score = 0
        for term in emphasis_terms:
            token = str(term).strip()
            if not token:
                continue
            if token in haystack:
                score += 2
            elif re.search(re.escape(token), haystack, re.IGNORECASE):
                score += 1
        return score

    def _extract_query_domain_hints(self, query: str) -> Dict[str, List[str]]:
        normalized = (query or "").replace(" ", "")
        hints = {
            "focus_terms": [],
            "exclude_terms": [],
        }

        if any(k in normalized for k in ["집", "주거", "주택", "전세", "월세", "임대", "보증금", "청약", "자취", "원룸"]):
            hints["focus_terms"] = ["주거", "주택", "전세", "월세", "임대", "보증금", "청약", "집"]

        if any(k in normalized for k in ["취업말고", "일자리말고", "구직말고", "취준말고"]):
            hints["exclude_terms"] = ["취업", "일자리", "구직", "인턴", "채용", "취준"]

        return hints

    def _is_news_like_query(self, query: str) -> bool:
        normalized = (query or "").replace(" ", "")
        return any(k in normalized for k in ["뉴스", "소식", "공지", "최신", "동향", "이슈"])

    def _score_domain_focus(self, result: PolicySearchResult, focus_terms: List[str], exclude_terms: List[str]) -> tuple:
        haystack = f"{result.title} {result.content} {result.category or ''}"
        focus_score = 0
        for term in focus_terms:
            if term and term in haystack:
                focus_score += 1

        exclude_hit = 0
        for term in exclude_terms:
            if term and term in haystack:
                exclude_hit = 1
                break

        # 우선순위: 제외의도 미충돌 > 도메인 일치 > 원래 검색 점수
        return (exclude_hit, -focus_score, -float(result.score or 0))

    def _resolve_content_image_url(self, metadata: Dict[str, Any]) -> str:
        source = str(metadata.get("source", ""))
        if source != "content":
            return ""

        content_id = str(metadata.get("content_id", "")).strip()
        if not content_id:
            raw_doc_id = str(metadata.get("policy_id", "")).strip()
            if raw_doc_id.startswith("content:"):
                content_id = raw_doc_id.split(":", 1)[1]

        if not content_id:
            return ""

        cached = self._content_image_url_cache.get(content_id)
        if cached is not None:
            return cached

        df = self._get_content_attachment_df()
        row = df[df["pstSn"].astype(str) == content_id]
        if row.empty:
            self._content_image_url_cache[content_id] = ""
            return ""

        attachment = str(row.iloc[0].get("atchFile", "")).strip()
        if not attachment:
            self._content_image_url_cache[content_id] = ""
            return ""

        if attachment.startswith("http://") or attachment.startswith("https://"):
            self._content_image_url_cache[content_id] = attachment
            return attachment

        if not attachment.startswith("data:image"):
            self._content_image_url_cache[content_id] = ""
            return ""

        try:
            prefix, encoded = attachment.split(",", 1)
            mime = "image/png"
            if ";base64" in prefix and ":" in prefix:
                mime = prefix.split(":", 1)[1].split(";", 1)[0]
            ext = self._mime_to_ext(mime)
            self._content_image_dir.mkdir(parents=True, exist_ok=True)
            filename = f"content_{content_id}.{ext}"
            file_path = self._content_image_dir / filename
            if not file_path.exists():
                file_path.write_bytes(base64.b64decode(encoded))
            url = f"/static/content_images/{filename}"
            self._content_image_url_cache[content_id] = url
            return url
        except (ValueError, binascii.Error, OSError):
            self._content_image_url_cache[content_id] = ""
            return ""
    
    def search(
        self,
        query: str,
        region: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        income_max: Optional[float] = None,
        emphasis_terms: Optional[List[str]] = None,
        k: int = 5,
    ) -> List[PolicySearchResult]:
        """
        정책 검색
        
        Args:
            query: 검색 쿼리
            region: 지역 필터
            age_min: 최소 나이
            age_max: 최대 나이
            income_max: 최대 소득
            k: 반환할 정책 수
        
        Returns:
            정책 검색 결과 리스트
        """
        # RAG 파이프라인 검색
        rag_results = self.rag_pipeline.search(
            query=query,
            k=k,
            use_reranker=True,
            region_filter=region,
            age_min=age_min,
            age_max=age_max,
            income_max=income_max,
        )

        # 프로필 필터가 너무 엄격해 결과가 없으면 완화 검색을 한 번 수행한다.
        if not rag_results and any(v is not None for v in [region, age_min, age_max, income_max]):
            rag_results = self.rag_pipeline.search(
                query=query,
                k=k,
                use_reranker=True,
                region_filter=None,
                age_min=None,
                age_max=None,
                income_max=None,
            )
        
        # 결과 변환
        results = []
        for result in rag_results:
            metadata = result.get("metadata", {})
            policy_result = PolicySearchResult(
                policy_id=metadata.get("policy_id", f"policy_{len(results)}"),
                title=metadata.get("policy_title", "미분류"),
                content=result["content"][:200],
                institution=metadata.get("institution", "?"),
                region=metadata.get("region", "?"),
                url=self._normalize_url(metadata.get("url", "")),
                score=result["score"],
                category=metadata.get("category", ""),
                source=metadata.get("source", "unknown"),
                image_url=self._resolve_content_image_url(metadata),
            )
            results.append(policy_result)

        region_token = self._extract_region_token(query)
        if region_token:
            results.sort(
                key=lambda r: 0 if region_token in f"{r.title} {r.content} {r.category or ''}" else 1
            )

        emphasis_terms = emphasis_terms or []
        if emphasis_terms:
            results.sort(
                key=lambda r: (-self._score_emphasis_match(r, emphasis_terms), -float(r.score or 0))
            )

        domain_hints = self._extract_query_domain_hints(query)
        if domain_hints["focus_terms"] or domain_hints["exclude_terms"]:
            results.sort(
                key=lambda r: self._score_domain_focus(
                    r,
                    domain_hints["focus_terms"],
                    domain_hints["exclude_terms"],
                )
            )

            # 제외 의도가 명확하면, 제외어 매칭 문서를 우선순위 뒤로 보내고 비매칭 문서를 먼저 반환한다.
            if domain_hints["exclude_terms"]:
                non_excluded = []
                excluded = []
                for r in results:
                    exclude_hit, _, _ = self._score_domain_focus(r, [], domain_hints["exclude_terms"])
                    if exclude_hit:
                        excluded.append(r)
                    else:
                        non_excluded.append(r)
                if non_excluded:
                    results = non_excluded + excluded

            # 도메인 집중어가 있으면 해당 문서를 상단에 모아 의도 불일치 노이즈를 줄인다.
            if domain_hints["focus_terms"]:
                focused = []
                others = []
                for r in results:
                    if self._score_emphasis_match(r, domain_hints["focus_terms"]) > 0:
                        focused.append(r)
                    else:
                        others.append(r)
                if focused:
                    results = focused + others

        # 일반 정책 질의에서는 content(뉴스/게시물)보다 policy 소스를 먼저 노출한다.
        if not self._is_news_like_query(query):
            results.sort(key=lambda r: 0 if str(r.source) == "policy" else 1)
        
        return results[:k]
    
    def invoke(self, input_data: PolicySearchInput) -> Dict[str, Any]:
        """
        LangChain Tool 인터페이스
        
        Args:
            input_data: PolicySearchInput
        
        Returns:
            도구 실행 결과
        """
        results = self.search(
            query=input_data.query,
            region=input_data.region,
            age_min=input_data.age_min,
            age_max=input_data.age_max,
            income_max=input_data.income_max,
            emphasis_terms=input_data.emphasis_terms,
            k=input_data.k,
        )
        
        return {
            "status": "success",
            "count": len(results),
            "policies": [
                {
                    "policy_id": r.policy_id,
                    "title": r.title,
                    "content": r.content,
                    "institution": r.institution,
                    "region": r.region,
                    "source": r.source,
                    "url": r.url,
                    "image_url": r.image_url,
                    "score": round(r.score, 3),
                    "category": r.category,
                }
                for r in results
            ]
        }


def policy_search_tool(rag_pipeline):
    """팩토리 함수"""
    return PolicySearchTool(rag_pipeline)
