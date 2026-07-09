"""슬롯 정규화 맵핑 테이블.

목적
- LLM/정규식이 뽑은 원시 값을 '정해진 표준값'으로만 매핑한다.
- 특히 나이는 검색 쿼리 텍스트에 넣지 않고 구조화 필터(age_min/age_max)로만 반환한다.
  (예: "약 20살"이 검색어에 섞여 유사도를 흐리는 문제 방지)

이 모듈은 결정론적이다. 여기서 매핑 실패한 값은 버리거나 None으로 둔다.
의미 확장(예: 적금 → 금융/저축/자산형성)은 시드만 제공하고,
최종 확장은 상위 단계의 선택 모델(LLM)이 담당한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from agent3.schemas.pipeline import (
    EmploymentStatus,
    HousingStatus,
    MarriageStatus,
    NormalizedSlots,
)


# ============================================================
# 지역: 표준 17개 시도로 수렴
# ============================================================

CANONICAL_REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

_REGION_ALIAS: dict[str, str] = {
    "서울특별시": "서울", "서울시": "서울",
    "부산광역시": "부산", "부산시": "부산",
    "대구광역시": "대구", "대구시": "대구",
    "인천광역시": "인천", "인천시": "인천",
    "광주광역시": "광주", "광주시": "광주",
    "대전광역시": "대전", "대전시": "대전",
    "울산광역시": "울산", "울산시": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원", "강원도": "강원",
    "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북",
    "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
    # 주요 시군 → 상위 시도
    "수원": "경기", "성남": "경기", "고양": "경기", "용인": "경기",
    "부천": "경기", "안산": "경기", "안양": "경기", "남양주": "경기",
    "화성": "경기", "평택": "경기", "의정부": "경기", "파주": "경기", "김포": "경기",
    "청주": "충북", "천안": "충남", "전주": "전북", "포항": "경북",
    "창원": "경남", "김해": "경남", "춘천": "강원", "원주": "강원",
}


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "", text)
    return text


def normalize_region(value: Optional[str]) -> Optional[str]:
    text = _clean(str(value or ""))
    if not text:
        return None
    if text in _REGION_ALIAS:
        return _REGION_ALIAS[text]
    if text in CANONICAL_REGIONS:
        return text
    for alias in sorted(_REGION_ALIAS, key=len, reverse=True):
        if alias in text:
            return _REGION_ALIAS[alias]
    for canonical in CANONICAL_REGIONS:
        if canonical in text:
            return canonical
    return None


# ============================================================
# 고용 / 결혼 / 주거: 별칭 → 표준 enum value
# ============================================================

_EMPLOYMENT_ALIAS: dict[str, str] = {
    "취업준비": EmploymentStatus.JOB_SEEKING.value,
    "취준": EmploymentStatus.JOB_SEEKING.value,
    "구직": EmploymentStatus.JOB_SEEKING.value,
    "재직": EmploymentStatus.EMPLOYED.value,
    "직장인": EmploymentStatus.EMPLOYED.value,
    "근무": EmploymentStatus.EMPLOYED.value,
    "실업": EmploymentStatus.UNEMPLOYED.value,
    "무직": EmploymentStatus.UNEMPLOYED.value,
    "학생": EmploymentStatus.STUDENT.value,
    "재학": EmploymentStatus.STUDENT.value,
    "대학생": EmploymentStatus.STUDENT.value,
}

_MARRIAGE_ALIAS: dict[str, str] = {
    "신혼": MarriageStatus.NEWLY_MARRIED.value,
    "기혼": MarriageStatus.MARRIED.value,
    "결혼": MarriageStatus.MARRIED.value,
    "부부": MarriageStatus.MARRIED.value,
    "미혼": MarriageStatus.SINGLE.value,
}

_HOUSING_ALIAS: dict[str, str] = {
    "무주택": HousingStatus.NO_HOUSE.value,
    "전세": HousingStatus.NO_HOUSE.value,
    "월세": HousingStatus.NO_HOUSE.value,
    "임차": HousingStatus.NO_HOUSE.value,
    "전월세": HousingStatus.NO_HOUSE.value,
    "자가": HousingStatus.HOME_OWNER.value,
    "자기집": HousingStatus.HOME_OWNER.value,
    "주택보유": HousingStatus.HOME_OWNER.value,
}


def _map_alias(text: str, table: dict[str, str]) -> Optional[str]:
    for alias, value in table.items():
        if alias in text:
            return value
    return None


# ============================================================
# 도메인 의미 키워드 시드 (LLM 확장의 출발점)
# ============================================================

DOMAIN_KEYWORD_SEED: dict[str, list[str]] = {
    "금융": ["금융", "저축", "적금", "예금", "자산형성", "목돈", "통장"],
    "주거": ["주거", "주택", "전세", "월세", "임대", "보증금", "청약", "기숙사"],
    "취업": ["취업", "일자리", "구직", "채용", "인턴", "일경험"],
    "창업": ["창업", "사업화", "스타트업", "창업지원"],
    "교육": ["교육", "훈련", "장학", "학자금", "역량"],
    "복지": ["복지", "생활지원", "심리", "건강", "돌봄"],
    "문화": ["문화", "여가", "예술", "체육"],
}

# 원시 단어 → 도메인 카테고리 (시드 확장을 위한 역인덱스)
_TERM_TO_DOMAIN: dict[str, str] = {}
for _domain, _terms in DOMAIN_KEYWORD_SEED.items():
    for _t in _terms:
        _TERM_TO_DOMAIN[_t] = _domain


def seed_domain_keywords(text: str) -> list[str]:
    """질문 텍스트에서 도메인 시드 키워드를 뽑는다(결정론적).

    최종 의미 확장은 상위 LLM이 담당하며, 여기서는 확실한 시드만 제공한다.
    """
    normalized = _clean(text)
    hits: list[str] = []
    seen: set[str] = set()
    for term, domain in _TERM_TO_DOMAIN.items():
        if term in normalized and domain not in seen:
            seen.add(domain)
            hits.extend(DOMAIN_KEYWORD_SEED[domain])
    # 중복 제거(순서 유지)
    return list(dict.fromkeys(hits))


# ============================================================
# 나이: 텍스트 → 구조화 필터 (쿼리 텍스트 오염 금지)
# ============================================================

_KOR_NUM = {
    "스무": 20, "스물": 20, "서른": 30, "마흔": 40,
}


def parse_age(text: str) -> dict[str, Optional[int]]:
    """나이 표현을 표준 구조화 값으로 변환.

    반환: {"age", "age_min", "age_max"}
    - "20살", "만 20세", "약 20살" → age=20, min=max=20
    - "20대 초반/중반/후반" → 범위
    - "스무살" → 20
    실패 시 모두 None.
    """
    result: dict[str, Optional[int]] = {"age": None, "age_min": None, "age_max": None}
    t = text or ""

    m = re.search(r"(?:만\s*)?(\d{1,2})\s*(?:세|살)", t)
    if m:
        age = int(m.group(1))
        if 0 < age < 100:
            result.update(age=age, age_min=age, age_max=age)
        return result

    for kor, val in _KOR_NUM.items():
        if kor in t and ("살" in t or "세" in t or "대" not in t):
            result.update(age=val, age_min=val, age_max=val)
            return result

    band = re.search(r"(\d{1,2})\s*대\s*(초반|중반|후반)?", t)
    if band:
        decade = int(band.group(1))
        if decade < 10:
            decade *= 10
        phase = band.group(2)
        if phase == "초반":
            lo, hi = decade, decade + 3
        elif phase == "중반":
            lo, hi = decade + 4, decade + 6
        elif phase == "후반":
            lo, hi = decade + 7, decade + 9
        else:
            lo, hi = decade, decade + 9
        result.update(age=None, age_min=lo, age_max=hi)
    return result


def parse_income(text: str) -> Optional[int]:
    """월 소득 표현을 원 단위 정수로. 단위 미표기는 '만원'으로 간주."""
    t = text or ""
    m = re.search(r"월\s*소득\s*(?:이|은|는)?\s*(\d{1,5})\s*(만원|만|원)?", t)
    if not m:
        m = re.search(r"소득\s*(?:이|은|는)?\s*(\d{1,5})\s*(만원|만|원)", t)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    if unit in ("만원", "만", None):
        return value * 10000
    return value


# ============================================================
# 통합 정규화기
# ============================================================

@dataclass
class SlotNormalizer:
    """원시 슬롯 dict를 NormalizedSlots(표준값)로 변환.

    입력 dict는 LLM 플래너가 뽑은 값 또는 텍스트에서 추출한 값 모두 허용.
    매핑 실패 값은 조용히 버려 검색 오염을 막는다.
    """

    def from_text(self, text: str) -> NormalizedSlots:
        age = parse_age(text)
        return NormalizedSlots(
            region=normalize_region(text),
            age=age["age"],
            age_min=age["age_min"],
            age_max=age["age_max"],
            income=parse_income(text),
            employment_status=_map_alias(_clean(text), _EMPLOYMENT_ALIAS),
            marriage_status=_map_alias(_clean(text), _MARRIAGE_ALIAS),
            housing_status=_map_alias(_clean(text), _HOUSING_ALIAS),
            domain_keywords=seed_domain_keywords(text),
        )

    def from_raw(self, raw: dict[str, Any]) -> NormalizedSlots:
        """LLM이 준 원시 슬롯 dict를 표준화."""
        text_region = raw.get("region")
        age_value = raw.get("age")
        age_min = raw.get("age_min")
        age_max = raw.get("age_max")

        if age_value is not None:
            try:
                age_value = int(age_value)
                age_min = age_max = age_value
            except (TypeError, ValueError):
                age_value = None

        return NormalizedSlots(
            region=normalize_region(text_region) if text_region else None,
            age=age_value,
            age_min=age_min,
            age_max=age_max,
            income=self._coerce_int(raw.get("income")),
            employment_status=self._normalize_enum(raw.get("employment_status"), _EMPLOYMENT_ALIAS, EmploymentStatus),
            marriage_status=self._normalize_enum(raw.get("marriage_status"), _MARRIAGE_ALIAS, MarriageStatus),
            children_count=self._coerce_int(raw.get("children_count")),
            housing_status=self._normalize_enum(raw.get("housing_status"), _HOUSING_ALIAS, HousingStatus),
            domain_keywords=[str(k).strip() for k in (raw.get("domain_keywords") or []) if str(k).strip()],
        )

    def merge(self, base: NormalizedSlots, override: NormalizedSlots) -> NormalizedSlots:
        """override 값이 있으면 우선(현재 질문 > 기억). None은 무시."""
        merged = NormalizedSlots(**base.__dict__)
        for key, value in override.__dict__.items():
            if key == "domain_keywords":
                merged.domain_keywords = list(dict.fromkeys([*base.domain_keywords, *override.domain_keywords]))
                continue
            if value not in (None, "", []):
                setattr(merged, key, value)
        return merged

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_enum(value: Any, alias_table: dict[str, str], enum_cls) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip()
        # 이미 표준 enum value면 그대로
        try:
            return enum_cls(text).value
        except ValueError:
            pass
        # 별칭 매핑
        return _map_alias(text, alias_table)
