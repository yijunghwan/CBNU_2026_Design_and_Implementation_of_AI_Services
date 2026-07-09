"""파이프라인 전 구간에서 공유하는 데이터 계약.

핵심 아이디어: 모든 단계는 PipelineContext를 받아 일부 필드를 채우고 반환한다.
단계 간 결합을 없애기 위해 각 단계의 산출물은 별도 dataclass로 명확히 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ============================================================
# 열거형 (표준값) - 맵핑 테이블 결과가 여기로 수렴한다
# ============================================================

class EmploymentStatus(str, Enum):
    JOB_SEEKING = "job_seeking"
    EMPLOYED = "employed"
    UNEMPLOYED = "unemployed"
    STUDENT = "student"


class MarriageStatus(str, Enum):
    SINGLE = "single"
    MARRIED = "married"
    NEWLY_MARRIED = "newly_married"


class HousingStatus(str, Enum):
    NO_HOUSE = "no_house"
    HOME_OWNER = "home_owner"


class MemoryAction(str, Enum):
    NONE = "none"
    ENABLE = "enable"
    DISABLE = "disable"
    SHOW_PROFILE = "show_profile"
    UPDATE_PROFILE = "update_profile"
    DELETE_PROFILE = "delete_profile"
    DELETE_HISTORY = "delete_history"
    DELETE_ALL = "delete_all"


class ToolName(str, Enum):
    RAG_POLICY_SEARCH = "rag_policy_search"
    RAG_CONTENT_SEARCH = "rag_content_search"
    WEB_SEARCH = "web_search"
    FILE_PARSE = "file_parse"


class JudgeDecision(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    CLARIFY = "clarify"


# ============================================================
# 슬롯: 질문/기억에서 추출 → 맵핑 테이블로 표준화된 값만 보관
# ============================================================

@dataclass
class NormalizedSlots:
    """검색 보조에 사용하는 표준화된 조건.

    주의: age 같은 숫자는 검색 쿼리 '텍스트'에 넣지 않는다.
    구조화 필터(age_min/age_max)로만 사용해 검색 오염을 막는다.
    """

    region: Optional[str] = None                 # 표준 시도 (예: 경기)
    age: Optional[int] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    income: Optional[int] = None                 # 월 소득(원)
    employment_status: Optional[str] = None      # EmploymentStatus value
    marriage_status: Optional[str] = None        # MarriageStatus value
    children_count: Optional[int] = None
    housing_status: Optional[str] = None         # HousingStatus value
    # 도메인 의미 키워드: 맵핑 테이블 시드 + 모델 의미확장 결과가 합쳐진다
    domain_keywords: list[str] = field(default_factory=list)

    def as_filter_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "age_min": self.age_min,
            "age_max": self.age_max,
            "income_max": self.income,
        }


# ============================================================
# 단계별 산출물
# ============================================================

@dataclass
class SecurityResult:
    passed: bool = True
    blocked_reason: Optional[str] = None
    sensitive_hits: list[str] = field(default_factory=list)


@dataclass
class ToolStep:
    """플래너가 계획한 개별 툴 실행 단위."""
    tool: str                                     # ToolName value
    reason: str = ""
    query: Optional[str] = None                   # 툴 전용 검색어(정규화 후 결정)


@dataclass
class PlannerDecision:
    """작은 LLM 플래너의 자율 결정 결과 (JSON 강제 파싱)."""
    intent: str = "general"                       # policy / term / latest / smalltalk / general ...
    is_policy_query: bool = False
    needs_user_profile: bool = False              # 개인 상황 기반 추천 여부

    # 메모리 제어
    memory_action: str = MemoryAction.NONE.value

    # 툴 계획
    can_answer_directly: bool = False             # 툴 없이 바로 답변 가능
    tool_plan: list[ToolStep] = field(default_factory=list)

    # 슬롯 충분성
    missing_slots: list[str] = field(default_factory=list)   # 부족 조건(사용자 안내용)
    domain_keywords: list[str] = field(default_factory=list) # 의미 확장 키워드

    raw: dict[str, Any] = field(default_factory=dict)         # 원본 LLM JSON(디버그)


@dataclass
class MemoryResult:
    """메모리 FSM 처리 결과. handled=True면 파이프라인 조기 종료."""
    handled: bool = False
    response_text: Optional[str] = None
    requires_confirmation: bool = False
    action: str = MemoryAction.NONE.value
    status: str = ""                              # needs_confirmation / completed / canceled ...


@dataclass
class ToolResults:
    """모든 툴 실행 결과 모음."""
    rag_policies: list[dict[str, Any]] = field(default_factory=list)
    rag_contents: list[dict[str, Any]] = field(default_factory=list)
    web_results: list[dict[str, Any]] = field(default_factory=list)
    file_text: Optional[str] = None
    executed_tools: list[str] = field(default_factory=list)
    # RAG 적합성 평가(이름/내용 유사도 포함)
    rag_evaluation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    decision: str = JudgeDecision.ACCEPT.value
    reason: str = ""
    refined_query: Optional[str] = None
    clarifying_question: Optional[str] = None


# ============================================================
# 컨텍스트 (메모리/요약 포함) — 모든 LLM 프롬프트 공통 재료
# ============================================================

@dataclass
class MemoryContext:
    """컨텍스트 윈도우 + 요약 + 장기 프로필."""
    short_window: list[dict[str, str]] = field(default_factory=list)   # 최근 대화(세션)
    short_summary: str = ""
    long_window: list[dict[str, str]] = field(default_factory=list)    # 장기기억 대화
    long_summary: str = ""
    long_profile: dict[str, Any] = field(default_factory=dict)         # 장기 저장 프로필
    long_memory_enabled: bool = False


@dataclass
class PipelineContext:
    """파이프라인 전 구간 공유 상태."""

    # ---- 입력 ----
    user_id: str
    query: str
    user_code: Optional[str] = None
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    file_path: Optional[str] = None
    file_type: Optional[str] = None

    # ---- 메모리/컨텍스트 ----
    memory: MemoryContext = field(default_factory=MemoryContext)

    # ---- 단계 산출물 ----
    security: SecurityResult = field(default_factory=SecurityResult)
    plan: PlannerDecision = field(default_factory=PlannerDecision)
    memory_result: MemoryResult = field(default_factory=MemoryResult)
    slots: NormalizedSlots = field(default_factory=NormalizedSlots)
    tools: ToolResults = field(default_factory=ToolResults)
    judge: JudgeResult = field(default_factory=JudgeResult)

    # ---- 최종 출력 ----
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None

    # ---- 관측성 ----
    trace: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

    # ---- 실시간 이벤트 훅 (스트리밍용, 직렬화 제외) ----
    on_event: Optional[Any] = field(default=None, repr=False, compare=False)

    def mark(self, stage: str) -> None:
        self.trace.append(stage)

    def emit(self, stage: str, phase: str, **extra: Any) -> None:
        if self.on_event:
            try:
                self.on_event({"stage": stage, "phase": phase, **extra})
            except Exception:  # noqa: BLE001
                pass
