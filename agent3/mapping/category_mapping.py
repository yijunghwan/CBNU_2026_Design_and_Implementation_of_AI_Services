"""분야(카테고리) 맵핑 테이블.

목적
- 질문 도메인과 정책 카테고리를 '같은 표준값'으로 고정해 하드조건 비교를 가능케 한다.
- 원본 카테고리(mclsfNm/lclsfNm)는 표기가 제각각이라 표준 6종으로 수렴시킨다.

표준 분야 (CANONICAL_FIELDS)
- 일자리   : 취업/창업/재직/일경험/인턴
- 주거     : 주택/전월세/기숙사/청약/임대
- 교육     : 미래역량강화/교육비/온오프라인교육/직업훈련/장학/자기개발
- 금융     : 금융지원/자산형성/적금/대출/저축
- 복지문화 : 문화활동/생활지원/건강/예술인지원/돌봄
- 참여권리 : 청년참여/정책인프라/권익보호/국제교류
"""

from __future__ import annotations

import re
from typing import Optional

CANONICAL_FIELDS = ["일자리", "주거", "교육", "금융", "복지문화", "참여권리"]

# 원본 카테고리/도메인 토큰 → 표준 분야
# (긴 토큰 우선 매칭을 위해 아래에서 길이순 정렬해 사용)
_FIELD_ALIAS: dict[str, str] = {
    # 일자리
    "일자리": "일자리", "취업": "일자리", "창업": "일자리", "재직": "일자리",
    "일경험": "일자리", "인턴": "일자리", "채용": "일자리", "구직": "일자리",
    # 주거
    "주거": "주거", "주택": "주거", "거주지": "주거", "전월세": "주거",
    "주거급여": "주거", "기숙사": "주거", "청약": "주거", "임대": "주거",
    "전세": "주거", "월세": "주거", "보증금": "주거",
    # 교육
    "교육": "교육", "직업훈련": "교육", "미래역량강화": "교육", "역량": "교육",
    "교육비": "교육", "온·오프라인교육": "교육", "온라인교육": "교육",
    "장학": "교육", "학자금": "교육", "자기개발": "교육", "자기계발": "교육",
    # 금융
    "금융": "금융", "금융지원": "금융", "취약계층 및 금융지원": "금융",
    "자산형성": "금융", "적금": "금융", "저축": "금융", "예금": "금융",
    "대출": "금융", "목돈": "금융",
    # 복지문화
    "복지문화": "복지문화", "복지": "복지문화", "문화활동": "복지문화",
    "문화활동 및 생활지원": "복지문화", "생활지원": "복지문화", "건강": "복지문화",
    "예술인지원": "복지문화", "예술": "복지문화", "돌봄": "복지문화", "심리": "복지문화",
    # 참여권리
    "참여권리": "참여권리", "참여기반": "참여권리", "청년참여": "참여권리",
    "정책인프라구축": "참여권리", "정책인프라": "참여권리", "권익보호": "참여권리",
    "청년국제교류": "참여권리", "국제교류": "참여권리",
}


def _clean(text: str) -> str:
    text = (text or "").strip()
    # 콤마 중복 표기(예: "취업,취업") 정리
    text = re.sub(r"\s+", " ", text)
    return text


def map_field(raw_category: Optional[str]) -> Optional[str]:
    """원본 카테고리 문자열을 표준 분야로. 실패 시 None."""
    text = _clean(str(raw_category or ""))
    if not text:
        return None
    # 정확 매칭 우선
    if text in _FIELD_ALIAS:
        return _FIELD_ALIAS[text]
    # 부분 매칭(긴 별칭 우선)
    for alias in sorted(_FIELD_ALIAS, key=len, reverse=True):
        if alias in text:
            return _FIELD_ALIAS[alias]
    return None


def map_field_from_many(*raw_values: Optional[str]) -> Optional[str]:
    """여러 후보(중분류 우선, 대분류 폴백)에서 첫 표준 분야를 찾는다."""
    for value in raw_values:
        field = map_field(value)
        if field:
            return field
    return None


def infer_query_field(domain_keywords: list[str], text: str = "") -> Optional[str]:
    """질문의 도메인 키워드(+원문)에서 표준 분야를 추론."""
    for kw in domain_keywords or []:
        field = map_field(kw)
        if field:
            return field
    return map_field(text)
