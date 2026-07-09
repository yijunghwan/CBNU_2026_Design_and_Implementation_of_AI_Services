"""
RAG 데이터 로더 - CSV/API 데이터 로드
"""

import os
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path
import re

from src.utils.region_mapper import normalize_region


class RAGDataLoader:
    """
    RAG용 데이터 로드 및 관리
    
    - CSV 파일에서 데이터 로드
    - API 데이터 추가 로드
    - 데이터 검증 및 정제
    """

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self._policy_meta_map: Optional[Dict[str, Dict[str, Any]]] = None

    def _to_int_or_none(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        try:
            num = float(text)
            if num <= 0:
                return None
            return int(num)
        except (TypeError, ValueError):
            return None

    def _to_float_or_none(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        try:
            num = float(text)
            if num <= 0:
                return None
            return num
        except (TypeError, ValueError):
            return None

    def _normalize_region_name(self, raw_region: Any) -> str:
        text = str(raw_region or "").strip()
        if not text or text.lower() == "nan":
            return ""
        normalized = normalize_region(text)
        return normalized or ""

    def _load_policy_meta_map(self) -> Dict[str, Dict[str, Any]]:
        if self._policy_meta_map is not None:
            return self._policy_meta_map

        policy_file = self.processed_dir / "youth_policy_processed.csv"
        if not policy_file.exists():
            self._policy_meta_map = {}
            return self._policy_meta_map

        try:
            df = pd.read_csv(policy_file, encoding="utf-8-sig", dtype=str)
        except Exception as e:
            print(f"⚠️ 정책 메타데이터 로드 실패: {e}")
            self._policy_meta_map = {}
            return self._policy_meta_map

        meta_map: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            policy_no = str(row.get("plcyNo", "")).strip()
            if not policy_no:
                continue

            region_raw = (
                row.get("operInstCdNm")
                or row.get("sprvsnInstCdNm")
                or row.get("rgtrInstCdNm")
                or ""
            )
            region = self._normalize_region_name(region_raw)

            meta_map[policy_no] = {
                "institution": str(row.get("operInstCdNm", "") or row.get("sprvsnInstCdNm", "") or "").strip(),
                "region": region,
                "eligibility_age_min": self._to_int_or_none(row.get("sprtTrgtMinAge")),
                "eligibility_age_max": self._to_int_or_none(row.get("sprtTrgtMaxAge")),
                "eligibility_income_max": self._to_float_or_none(row.get("earnMaxAmt")),
                "metadata_version": "v2_policy_enriched",
            }

        self._policy_meta_map = meta_map
        print(f"✅ 정책 메타데이터 매핑 로드: {len(meta_map)} 개")
        return self._policy_meta_map

    def load_rag_documents(self) -> List[Dict[str, Any]]:
        """
        RAG 문서 로드 (이미 전처리된 CSV)
        
        Returns:
            문서 리스트 [{"id": "", "content": "", "metadata": {}}, ...]
        """
        rag_file = self.processed_dir / "youth_rag_documents.csv"
        
        if not rag_file.exists():
            print(f"⚠️ RAG 문서 파일 없음: {rag_file}")
            return []
        
        try:
            df = pd.read_csv(rag_file, encoding='utf-8-sig')
            print(f"✅ RAG 문서 로드: {len(df)} 개")
            policy_meta_map = self._load_policy_meta_map()
            
            documents = []
            for idx, row in df.iterrows():
                source = str(row.get("source", "unknown"))
                doc_id = str(row.get("doc_id", f"doc_{idx}"))
                content_id = ""
                if source == "content" and ":" in doc_id:
                    content_id = doc_id.split(":", 1)[1]

                # CSV 컬럼명 매핑
                policy_meta: Dict[str, Any] = {}
                if source == "policy" and ":" in doc_id:
                    policy_no = doc_id.split(":", 1)[1].strip()
                    policy_meta = policy_meta_map.get(policy_no, {})

                doc = {
                    "id": doc_id,
                    "content": str(row.get("text", "")),  # "text" 컬럼 사용
                    "metadata": {
                        "source": source,
                        "policy_id": doc_id,
                        "policy_title": str(row.get("title", "")),
                        "title": str(row.get("title", "")),
                        "category": str(row.get("category", "")),
                        "url": str(row.get("url", "")),
                        "updated_at": str(row.get("updated_at", "")),
                        "content_id": content_id,
                        "institution": policy_meta.get("institution", ""),
                        "region": policy_meta.get("region", ""),
                        "eligibility_age_min": policy_meta.get("eligibility_age_min"),
                        "eligibility_age_max": policy_meta.get("eligibility_age_max"),
                        "eligibility_income_max": policy_meta.get("eligibility_income_max"),
                        "metadata_version": policy_meta.get("metadata_version", "v1"),
                    }
                }
                documents.append(doc)
            
            return documents
        
        except Exception as e:
            print(f"❌ RAG 문서 로드 실패: {e}")
            return []

    def load_policies(self) -> List[Dict[str, Any]]:
        """정책 데이터 로드"""
        policy_file = self.processed_dir / "youth_policy.csv"
        
        if not policy_file.exists():
            print(f"⚠️ 정책 파일 없음: {policy_file}")
            return []
        
        try:
            df = pd.read_csv(policy_file, encoding='utf-8-sig')
            print(f"✅ 정책 로드: {len(df)} 개")
            return df.to_dict('records')
        except Exception as e:
            print(f"❌ 정책 로드 실패: {e}")
            return []

    def load_content(self) -> List[Dict[str, Any]]:
        """콘텐츠 데이터 로드"""
        content_file = self.processed_dir / "youth_content.csv"
        
        if not content_file.exists():
            print(f"⚠️ 콘텐츠 파일 없음: {content_file}")
            return []
        
        try:
            df = pd.read_csv(content_file, encoding='utf-8-sig')
            print(f"✅ 콘텐츠 로드: {len(df)} 개")
            return df.to_dict('records')
        except Exception as e:
            print(f"❌ 콘텐츠 로드 실패: {e}")
            return []

    def get_data_stats(self) -> Dict[str, Any]:
        """데이터 통계"""
        rag_docs = self.load_rag_documents()
        policies = self.load_policies()
        content = self.load_content()
        
        return {
            "rag_documents": len(rag_docs),
            "policies": len(policies),
            "content": len(content),
            "total_documents": len(rag_docs),
        }


def load_rag_data(data_dir: str = "data") -> List[Dict[str, Any]]:
    """RAG 데이터 로드 (편의 함수)"""
    loader = RAGDataLoader(data_dir)
    return loader.load_rag_documents()
