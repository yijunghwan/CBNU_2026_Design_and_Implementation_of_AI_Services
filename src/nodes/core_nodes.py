"""
LangGraph 핵심 노드들

1. 미들웨어 체크 노드
2. 라우팅 노드
3. 도구 호출 노드
4. 응답 생성 노드
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
import re
from types import SimpleNamespace
from time import perf_counter

from src.middleware.middleware_chain import MiddlewareChain
from src.schemas.state import AgentState
from src.tools import create_tool_registry
from src.rag.pipeline import create_rag_pipeline
from src.services.node_llm_service import NodeLLMService
from src.services.profile_inference_service import ProfileInferenceService
from src.services.tool_orchestration_service import ToolOrchestrationService

logger = logging.getLogger(__name__)

_RAG_PIPELINE = None
_TOOL_REGISTRY = None
_NODE_LLM_SERVICE = NodeLLMService()
_PROFILE_INFERENCE_SERVICE = ProfileInferenceService()
_TOOL_ORCHESTRATION_SERVICE = ToolOrchestrationService()

_GRAPH_NODE_ORDER = [
    "middleware_check",
    "routing",
    "tool_calling",
    "retrieval_evaluation",
    "response_generation",
    "response_evaluation",
    "output",
]

_GRAPH_EDGES = [
    {"from": "middleware_check", "to": "routing", "label": "passed"},
    {"from": "middleware_check", "to": "user_action_required", "label": "consent/deletion"},
    {"from": "middleware_check", "to": "error", "label": "security_warning"},
    {"from": "routing", "to": "tool_calling", "label": "action_selected"},
    {"from": "tool_calling", "to": "retrieval_evaluation", "label": "tools_done"},
    {"from": "retrieval_evaluation", "to": "response_generation", "label": "evaluated"},
    {"from": "response_generation", "to": "response_evaluation", "label": "drafted"},
    {"from": "response_evaluation", "to": "tool_calling", "label": "retry"},
    {"from": "response_evaluation", "to": "output", "label": "accepted_or_clarified"},
    {"from": "output", "to": "END", "label": "return"},
]


def _build_node_debug(
    state: AgentState,
    node_name: str,
    node_start_time: float,
    extra_debug: Optional[Dict[str, Any]] = None,
) -> dict:
    debug_info = dict(state.debug_info or {})
    trace = list(debug_info.get("node_trace") or [])
    trace.append(node_name)
    timings = dict(debug_info.get("node_timings_ms") or {})
    timings[node_name] = round((perf_counter() - node_start_time) * 1000, 2)
    debug_info["node_trace"] = trace
    debug_info["node_timings_ms"] = timings
    if extra_debug:
        debug_info.update(extra_debug)
    return debug_info


def _build_path_edges(path_nodes: list[str]) -> list[dict]:
    edges: list[dict] = []
    if not path_nodes:
        return edges
    for idx in range(len(path_nodes) - 1):
        edges.append({"from": path_nodes[idx], "to": path_nodes[idx + 1]})
    return edges


def _synthesize_web_answer_with_llm(state: AgentState, web_result: dict) -> Optional[str]:
    """웹 검색 결과를 LLM으로 요약/정리해 최종 답변 생성"""
    return _NODE_LLM_SERVICE.synthesize_web_answer(state, web_result)


def _synthesize_general_answer_with_llm(state: AgentState) -> Optional[str]:
    """도구 호출이 필요 없는 일반 질문을 내부 LLM으로 응답 생성"""
    return _NODE_LLM_SERVICE.synthesize_general_answer(state)


def _synthesize_policy_answer_with_llm(state: AgentState, tool_results: dict) -> Optional[str]:
    """policy_search/web_search 결과를 참고해 하나의 최종 답변으로 합성"""
    profile = _build_effective_profile(state)
    return _NODE_LLM_SERVICE.synthesize_policy_answer(state, tool_results, profile)


def _has_actionable_policy_constraints(text: str) -> bool:
    normalized = (text or "").replace(" ", "")
    domain_tokens = [
        "정책", "지원", "금융", "금용", "적금", "저축", "예금", "대출", "청약", "주거", "취업",
    ]
    has_domain = any(token in normalized for token in domain_tokens)

    extracted = _extract_profile_from_text(text or "")
    has_age = bool(extracted.get("age") or extracted.get("age_band") or (extracted.get("age_min") and extracted.get("age_max")))
    has_region = bool(extracted.get("region") or _extract_query_region_intent(text or ""))

    return has_domain and (has_age or has_region)


def _build_small_talk_message() -> str:
    return (
        "안녕하세요! 저는 청년 정책 추천 에이전트예요.\n"
        "원하시면 '내 상황에 맞는 정책 추천해줘'처럼 바로 질문해 주세요."
    )


def _build_agent_intro_message() -> str:
    return (
        "안녕하세요! 저는 청년 정책 추천 에이전트예요.\n"
        "제가 도와드릴 수 있는 기능은 다음과 같아요:\n"
        "1) 정책 검색/추천: 조건(나이, 지역, 소득, 고용상태)에 맞는 정책 안내\n"
        "2) 추천 근거 설명: 왜 이 정책을 추천했는지 기준 설명\n"
        "3) 장기기억(선택): 동의 후 사용자 정보/질문 맥락 기억\n"
        "4) 파일 분석: 업로드 문서/이미지에서 텍스트 추출\n"
        "\n"
        "원하시면 이렇게 질문해보세요: '만 26세 서울 거주인데 받을 수 있는 정책 추천해줘'"
    )


def _is_self_name_question(text: str) -> bool:
    normalized = text.replace(" ", "")
    patterns = ["내이름이뭐", "내이름뭐", "제이름이뭐", "이름이뭐야", "내이름뭐야", "제이름뭐야"]
    return any(p in normalized for p in patterns)


def _is_previous_question_query(text: str) -> bool:
    normalized = text.replace(" ", "")
    patterns = [
        "이전에내가했던질문이뭐지", "이전에내질문이뭐지", "직전질문이뭐지", "방금내가한질문이뭐지",
        "내가아까뭐라고했지", "이전에뭐물어봤지", "지난질문이뭐였지",
        "이전질문", "그이전질문", "그전질문", "그이전", "그전",
    ]
    return any(p in normalized for p in patterns)


def _extract_previous_question_offset(text: str) -> int:
    normalized = (text or "").replace(" ", "")
    # "그 이전 질문"은 직전 질문보다 한 단계 더 이전 질문을 의미하도록 해석한다.
    if any(p in normalized for p in ["그이전질문", "그전질문", "그이전", "그전", "전전질문"]):
        return 2
    return 1


def _is_conversation_recap_query(text: str) -> bool:
    normalized = (text or "").replace(" ", "")
    patterns = [
        "지금까지대화내용", "지금까지대화", "대화내용말해", "대화요약", "대화정리",
        "우리가무슨얘기", "무슨대화했", "이전화내용정리", "대화기록요약",
    ]
    return any(p in normalized for p in patterns)


def _find_previous_user_question(state: AgentState) -> Optional[str]:
    merged_history = []
    merged_history.extend(state.conversation_history or [])
    merged_history.extend((state.debug_info or {}).get("long_term_history", []))

    offset = _extract_previous_question_offset(state.user_input)

    seen = set()
    candidates = []
    for message in reversed(merged_history):
        if message.get("role") != "user":
            continue
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if content == (state.user_input or "").strip():
            continue
        if content in seen:
            continue
        seen.add(content)
        candidates.append(content)

    if len(candidates) >= offset:
        return candidates[offset - 1]
    return None


def _extract_user_name(text: str) -> Optional[str]:
    if not text:
        return None

    match = re.search(r"(?:내\s*이름은|제\s*이름은)\s*([가-힣A-Za-z]{2,20})", text)
    if match:
        raw = match.group(1).strip()
        suffixes = ["입니다", "이에요", "예요", "이야", "야"]
        for suffix in suffixes:
            if raw.endswith(suffix) and len(raw) > len(suffix) + 1:
                raw = raw[: -len(suffix)]
                break
        return raw.strip()

    # 구어체 자기소개도 허용 (예: "이정환이야", "홍길동입니다", "이정환이야 그")
    colloquial = re.search(r"\b([가-힣A-Za-z]{2,20})\s*(?:이야|야|입니다|이에요|예요)(?:\s|$|[.!?~])", text)
    if colloquial:
        return colloquial.group(1).strip()

    return None


def _is_name_statement(text: str) -> bool:
    if not text:
        return False
    if bool(re.search(r"(?:내\s*이름은|제\s*이름은)", text)) and bool(_extract_user_name(text)):
        return True
    return bool(re.search(r"\b[가-힣A-Za-z]{2,20}\s*(?:이야|야|입니다|이에요|예요)(?:\s|$|[.!?~])", text))


def _find_user_name_from_history(history: list[dict[str, str]]) -> Optional[str]:
    for message in reversed(history or []):
        if message.get("role") != "user":
            continue
        name = _extract_user_name(message.get("content", ""))
        if name:
            return name
    return None


def _find_user_name_from_state(state: AgentState) -> Optional[str]:
    short = _find_user_name_from_history(state.conversation_history or [])
    if short:
        return short
    long_history = (state.debug_info or {}).get("long_term_history", [])
    return _find_user_name_from_history(long_history)


def _build_conversation_recap(state: AgentState, max_items: int = 6) -> str:
    merged = []
    merged.extend(state.conversation_history or [])
    merged.extend((state.debug_info or {}).get("long_term_history", []))

    seen = set()
    user_items = []
    for msg in reversed(merged):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if content == (state.user_input or "").strip():
            continue
        if content in seen:
            continue
        seen.add(content)
        user_items.append(content)
        if len(user_items) >= max_items:
            break

    if not user_items:
        return "아직 요약할 이전 대화가 없어요."

    items = list(reversed(user_items))
    lines = ["지금까지 대화에서 사용자가 주로 말한 내용은 아래와 같아요."]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def _needs_web_complement_for_policy(query: str, policy_count: int) -> bool:
    if policy_count < 3:
        return True
    complement_keywords = ["최신", "최근", "뉴스", "마감", "공고", "신청기간", "변경"]
    return any(keyword in query for keyword in complement_keywords)


def _is_web_search_needed_query(text: str) -> bool:
    normalized = (text or "").replace(" ", "")
    web_keywords = [
        "웹검색", "인터넷검색", "최신", "최근", "속보", "방금", "오늘", "뉴스", "마감", "공고", "변경",
    ]
    return any(k in normalized for k in web_keywords)


def _classify_query_intent_with_llm(state: AgentState) -> dict:
    """키워드 단순 매칭 대신 질문 의도를 LLM으로 분류한다."""
    fallback = {
        # LLM 실패 시에도 키워드 강제분기를 피하기 위해 중립 기본값을 사용한다.
        "is_policy_query": False,
        "is_profile_recommendation": False,
        "is_term_query": False,
        "needs_web_search": False,
        "exclude_policy": False,
        "use_user_profile": False,
        "emphasis_terms": [],
    }

    return _NODE_LLM_SERVICE.classify_query_intent(state, fallback)


def _get_intent(state: AgentState) -> dict:
    cached = (state.debug_info or {}).get("intent")
    return cached if isinstance(cached, dict) else _classify_query_intent_with_llm(state)


def _should_use_user_profile(state: AgentState) -> bool:
    intent = _get_intent(state)
    return bool(intent.get("use_user_profile"))


def _is_recommendation_basis_question(text: str) -> bool:
    normalized = text.replace(" ", "")
    keywords = ["뭘기반", "무슨기준", "추천이유", "왜추천", "기준으로", "근거"]
    return any(k in normalized for k in keywords)


def _is_memory_based_recommendation_request(text: str) -> bool:
    normalized = (text or "").replace(" ", "")
    memory_signals = ["장기기억", "개인정보기록", "개인정보", "기억", "기반"]
    recommendation_signals = ["추천", "정책", "맞는", "맞춤", "이용하면", "되잖", "써서"]
    return any(k in normalized for k in memory_signals) and any(k in normalized for k in recommendation_signals)


def _has_recommendation_profile(profile: dict) -> bool:
    if not profile:
        return False
    has_age_hint = bool(profile.get("age") or profile.get("age_band") or (profile.get("age_min") and profile.get("age_max")))
    signal_count = sum(
        1 for key in [
            "region", "income", "employment_status",
            "marriage_status", "children_count", "housing_status",
        ] if profile.get(key)
    )
    signal_count += 1 if has_age_hint else 0
    return signal_count >= 2


def _extract_profile_from_text(text: str) -> dict:
    return _PROFILE_INFERENCE_SERVICE.extract_profile_from_text(text)


def _extract_query_region_intent(text: str) -> Optional[str]:
    return _PROFILE_INFERENCE_SERVICE.extract_query_region_intent(text)


def _build_effective_profile(state: AgentState, include_memory_profile: bool = True) -> dict:
    return _PROFILE_INFERENCE_SERVICE.build_effective_profile(
        state,
        include_memory_profile=include_memory_profile,
    )


def _build_active_profile(state: AgentState) -> dict:
    intent = _get_intent(state)
    should_use_memory_profile = _should_use_user_profile(state)

    # 의도 분류가 보수적으로 나와도, 저장된 장기 프로필이 있고
    # 문장이 개인 맞춤/기반 추천을 강하게 시사하면 메모리 프로필을 반영한다.
    if not should_use_memory_profile:
        normalized = (state.user_input or "").replace(" ", "")
        personalized_tokens = ["나에게", "내게", "맞춤", "기반", "내조건", "내상황", "나한테"]
        has_personalized_signal = any(token in normalized for token in personalized_tokens)
        has_recommendation_signal = bool(intent.get("is_profile_recommendation")) or any(
            token in normalized for token in ["추천", "정책", "주택", "주거"]
        )
        long_profile = (state.debug_info or {}).get("long_term_user_profile") or {}
        has_long_profile = any(long_profile.get(k) for k in ["age", "region", "income", "employment_status", "housing_status"])
        should_use_memory_profile = has_personalized_signal and has_recommendation_signal and has_long_profile

    return _build_effective_profile(state, include_memory_profile=should_use_memory_profile)


def _extract_profile_from_memories(state: AgentState) -> dict:
    return _PROFILE_INFERENCE_SERVICE.extract_profile_from_memories(state)


def _build_profile_gap_message(profile: dict) -> str:
    return _PROFILE_INFERENCE_SERVICE.build_profile_gap_message(profile)


def _build_recommendation_basis(state: AgentState, profile_used: Optional[dict] = None) -> list[str]:
    basis = [
        f"질문 키워드 기반 의미 검색: '{state.user_input}'",
        "청년정책 RAG 벡터 검색 유사도 점수 상위 정책 우선",
    ]
    profile = profile_used or _build_effective_profile(state)
    if profile:
        if profile.get("region"):
            basis.append(f"지역 조건 반영: {profile.get('region')}")
        if profile.get("age"):
            basis.append(f"연령 조건 반영: {profile.get('age')}세")
        if profile.get("income"):
            basis.append("소득 조건 반영")
        if profile.get("employment_status"):
            basis.append(f"고용 상태 반영: {profile.get('employment_status')}")
        if profile.get("marriage_status"):
            basis.append(f"결혼 여부 반영: {profile.get('marriage_status')}")
        if profile.get("children_count") is not None:
            basis.append(f"자녀 수 반영: {profile.get('children_count')}명")
        if profile.get("housing_status"):
            basis.append(f"주거 상태 반영: {profile.get('housing_status')}")
        if profile.get("household_role"):
            basis.append(f"세대 역할 반영: {profile.get('household_role')}")
        if profile.get("pregnancy_status"):
            basis.append("임신/출산 상태 반영")
        if profile.get("military_service_status"):
            basis.append(f"병역 상태 반영: {profile.get('military_service_status')}")
    return basis


def _augment_policy_query_with_profile(query: str, profile: dict) -> str:
    return _PROFILE_INFERENCE_SERVICE.augment_policy_query_with_profile(query, profile)


def _get_tool_registry():
    """RAG 파이프라인 포함 도구 레지스트리 지연 초기화"""
    global _RAG_PIPELINE, _TOOL_REGISTRY

    if _TOOL_REGISTRY is not None:
        return _TOOL_REGISTRY

    rag_pipeline = None
    try:
        _RAG_PIPELINE = create_rag_pipeline(
            data_dir="data",
            persist_dir="data/vectorstore",
        )
        raw_docs = _RAG_PIPELINE.load_data()
        _RAG_PIPELINE.chunk_documents(raw_docs)

        vs_stats = _RAG_PIPELINE.vectorstore.get_stats()
        doc_count = vs_stats.get("document_count", 0)
        if doc_count == 0:
            _RAG_PIPELINE.build_vectorstore(rebuild=False)
        elif _is_vectorstore_metadata_outdated(_RAG_PIPELINE, expected_version="v2_policy_enriched"):
            logger.info("Outdated vectorstore metadata detected. Rebuilding for enriched policy metadata.")
            _RAG_PIPELINE.build_vectorstore(rebuild=True)

        _RAG_PIPELINE.build_retriever(alpha=0.5)
        _RAG_PIPELINE.build_reranker()
        rag_pipeline = _RAG_PIPELINE
        logger.info("RAG pipeline initialized for tool registry")
    except Exception as e:
        logger.warning(f"RAG pipeline initialization skipped: {str(e)}")

    _TOOL_REGISTRY = create_tool_registry(rag_pipeline=rag_pipeline, db_manager=None)
    return _TOOL_REGISTRY


def _is_vectorstore_metadata_outdated(rag_pipeline, expected_version: str) -> bool:
    try:
        vs = rag_pipeline.vectorstore.vectorstore
        if not vs or not hasattr(vs, "_collection"):
            return False
        sample = vs._collection.get(limit=1, include=["metadatas"])
        metadatas = (sample or {}).get("metadatas") or []
        if not metadatas:
            return True
        first_meta = metadatas[0] or {}
        return str(first_meta.get("metadata_version", "v1")) != expected_version
    except Exception as e:
        logger.debug(f"vectorstore metadata version check skipped: {str(e)}")
        return False


def _tool_input(**kwargs):
    return SimpleNamespace(**kwargs)


# ============ 1. 미들웨어 체크 노드 ============

def middleware_check_node(state: AgentState) -> dict:
    """
    4중 미들웨어 체인 실행
    
    Returns:
        상태 업데이트 딕셔너리
    """
    print(f"\n[NODE] middleware_check_node")
    print(f"  User input: {state.user_input[:100]}...")
    node_start = perf_counter()

    passed, updated_state, action_type = MiddlewareChain.execute(state)

    if not passed:
        print(f"  ⚠️ Middleware check failed: {action_type}")
        return {
            "current_stage": "middleware_check",
            "privacy_consent_requested": updated_state.privacy_consent_requested,
            "deletion_confirm_requested": updated_state.deletion_confirm_requested,
            "security_warning": updated_state.security_warning,
            "debug_info": _build_node_debug(state, "middleware_check", node_start),
        }

    print(f"  ✅ All middleware checks passed")
    return {
        "current_stage": "routing",
        "debug_info": _build_node_debug(state, "middleware_check", node_start),
    }


# ============ 2. 라우팅 노드 (조건부 분기) ============

def routing_node(state: AgentState) -> dict:
    """
    사용자 입력을 분석하여 에이전트 액션 결정
    
    가능한 액션:
    - profile_matching: 프로필 기반 정책 추천
    - general_search: 일반 정책 검색
    - bookmark_management: 북마크 관리
    - data_deletion: 데이터 삭제
    - data_refresh: 데이터 새로고침
    
    Returns:
        선택된 액션
    """
    print(f"\n[NODE] routing_node")
    node_start = perf_counter()

    intent = _classify_query_intent_with_llm(state)
    fallback_action = "general_search"

    if intent.get("is_profile_recommendation") and not intent.get("exclude_policy"):
        fallback_action = "profile_matching"

    action = _NODE_LLM_SERVICE.classify_agent_action(state, fallback_action)

    # LLM 분류가 빗나가더라도 정책 제외 의도는 강제 반영한다.
    if intent.get("exclude_policy") and action in ["profile_matching"]:
        action = "general_search"

    print(f"  → Action: {action} (LLM 기반 라우팅)")

    return {
        "agent_action": action,
        "debug_info": _build_node_debug(state, "routing", node_start, {"intent": intent}),
        "current_stage": "tool_calling",
    }


# ============ 3. 도구 호출 노드 ============

def tool_calling_node(state: AgentState) -> dict:
    """
    선택된 액션에 따라 도구 호출 - ToolRegistry 사용
    
    도구 목록:
    - multimodal_parser: 멀티모달 파서
    - web_search: 웹 검색
    - policy_search: 정책 검색 (RAG)
    - bookmark: 북마크 관리
    - data_deletion: 데이터 삭제
    
    Returns:
        도구 호출 결과
    """
    print(f"\n[NODE] tool_calling_node")
    print(f"  Agent action: {state.agent_action}")
    node_start = perf_counter()

    # 도구 레지스트리 초기화 (RAG 지연 초기화 포함)
    registry = _get_tool_registry()
    
    try:
        effective_query = str((state.debug_info or {}).get("retry_refined_query") or state.user_input or "")
        llm_conv_intent = _NODE_LLM_SERVICE.classify_conversation_intent(
            state,
            fallback={
                # 트리거 기반 보정보다 LLM 판정을 우선하기 위해 보수적 fallback 사용
                "is_agent_capability_query": False,
                "is_small_talk_query": False,
                "is_recommendation_basis_question": False,
                "is_memory_based_recommendation_request": False,
                "is_name_statement": False,
                "is_previous_question_query": False,
                "is_conversation_recap_query": False,
                "is_self_name_question": False,
            },
        )

        result = _TOOL_ORCHESTRATION_SERVICE.execute(
            state=state,
            query_text=effective_query,
            conversation_intent=llm_conv_intent,
            registry=registry,
            tool_input_builder=_tool_input,
            build_agent_intro_message=_build_agent_intro_message,
            build_small_talk_message=_build_small_talk_message,
            has_recommendation_profile=_has_recommendation_profile,
            build_active_profile=_build_active_profile,
            get_intent=_get_intent,
            augment_policy_query_with_profile=_augment_policy_query_with_profile,
            extract_user_name=_extract_user_name,
            find_previous_user_question=_find_previous_user_question,
            build_conversation_recap=_build_conversation_recap,
            find_user_name_from_state=_find_user_name_from_state,
            extract_query_region_intent=_extract_query_region_intent,
            build_profile_gap_message=_build_profile_gap_message,
            needs_web_complement_for_policy=_needs_web_complement_for_policy,
            synthesize_general_answer_with_llm=_synthesize_general_answer_with_llm,
            select_policy_search_plan=lambda query, intent: _NODE_LLM_SERVICE.select_policy_search_plan(state, query, intent),
        )
        result.setdefault("debug_info", {})
        result["debug_info"]["effective_query"] = effective_query
        result["debug_info"]["conversation_intent"] = llm_conv_intent
        result["debug_info"] = _build_node_debug(
            state,
            "tool_calling",
            node_start,
            result.get("debug_info") or {},
        )
        return result

    except Exception as e:
        logger.error(f"Tool calling error: {str(e)}")
        return {
            "tool_calls": [],
            "tool_results": {
                "error": {
                    "status": "error",
                    "message": f"도구 호출 실패: {str(e)}"
                }
            },
            "current_stage": "retrieval_evaluation",
            "debug_info": _build_node_debug(state, "tool_calling", node_start),
        }


def _extract_retrieval_eval_hints(query: str, intent: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized = (query or "").replace(" ", "")
    focus_terms = list(intent.get("emphasis_terms") or [])
    exclude_terms: List[str] = []

    if any(k in normalized for k in ["집", "주거", "주택", "전세", "월세", "임대", "보증금", "청약", "자취", "원룸"]):
        focus_terms.extend(["주거", "주택", "전세", "월세", "임대", "보증금", "청약", "집"])

    if any(k in normalized for k in ["취업말고", "일자리말고", "구직말고", "취준말고"]):
        exclude_terms = ["취업", "일자리", "구직", "인턴", "채용", "취준"]

    # 중복 제거
    focus_terms = [t for t in dict.fromkeys([str(t).strip() for t in focus_terms if str(t).strip()])]
    return {
        "focus_terms": focus_terms,
        "exclude_terms": exclude_terms,
    }


def _evaluate_policy_relevance(
    *,
    query: str,
    intent: Dict[str, Any],
    policy_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not policy_results:
        return {
            "status": "poor",
            "score": 0.0,
            "summary": "정책 검색 결과가 없습니다.",
            "focus_match_ratio": 0.0,
            "exclude_clean_ratio": 0.0,
            "policy_source_ratio": 0.0,
            "top_n": 0,
        }

    hints = _extract_retrieval_eval_hints(query, intent)
    focus_terms = hints["focus_terms"]
    exclude_terms = hints["exclude_terms"]
    top = policy_results[:10]
    top_n = len(top)

    focus_hits = 0
    exclude_hits = 0
    policy_source_hits = 0

    for item in top:
        haystack = f"{item.get('title', '')} {item.get('content', '')} {item.get('category', '')}"
        if focus_terms and any(term in haystack for term in focus_terms):
            focus_hits += 1
        if exclude_terms and any(term in haystack for term in exclude_terms):
            exclude_hits += 1
        if str(item.get("source", "")) == "policy":
            policy_source_hits += 1

    focus_ratio = (focus_hits / top_n) if top_n else 0.0
    exclude_clean_ratio = (1 - (exclude_hits / top_n)) if top_n else 0.0
    policy_source_ratio = (policy_source_hits / top_n) if top_n else 0.0

    # 질의 의도 기반 가중 점수
    score = 0.0
    if focus_terms:
        score += focus_ratio * 0.5
    else:
        score += 0.2

    if exclude_terms:
        score += exclude_clean_ratio * 0.3
    else:
        score += 0.2

    score += policy_source_ratio * 0.2

    if score >= 0.62:
        status = "good"
    elif score >= 0.4:
        status = "borderline"
    else:
        status = "poor"

    return {
        "status": status,
        "score": round(score, 3),
        "summary": "질의-정책 적합성 평가 완료",
        "focus_match_ratio": round(focus_ratio, 3),
        "exclude_clean_ratio": round(exclude_clean_ratio, 3),
        "policy_source_ratio": round(policy_source_ratio, 3),
        "top_n": top_n,
    }


def retrieval_evaluation_node(state: AgentState) -> dict:
    """
    정책 검색 결과를 질문 의도와 대조해 적합성 평가.
    품질이 낮으면 web_search를 보완 호출한다.
    """
    print(f"\n[NODE] retrieval_evaluation_node")
    node_start = perf_counter()

    tool_results = dict(state.tool_results or {})
    tool_calls = list(state.tool_calls or [])
    intent = _get_intent(state)
    should_policy_guard = bool(intent.get("is_policy_query") or intent.get("is_term_query")) and not bool(intent.get("exclude_policy"))

    # 정책/용어 질의가 아니면 검색 품질 평가를 건너뛰어
    # 일반 대화(예: 이름 소개)에 정책 부족 안내가 섞이지 않도록 한다.
    if not should_policy_guard:
        tool_results["retrieval_evaluation"] = {
            "status": "skipped",
            "score": None,
            "summary": "정책 질의가 아니어서 검색 품질 평가를 생략",
            "web_fallback_triggered": False,
        }
        return {
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "current_stage": "response_generation",
            "debug_info": _build_node_debug(state, "retrieval_evaluation", node_start),
        }

    if "profile_needed" in tool_results:
        tool_results["retrieval_evaluation"] = {
            "status": "needs_profile",
            "score": 0.0,
            "summary": "추천을 위한 최소 사용자 정보가 부족하여 검색 품질 평가를 건너뜀",
            "web_fallback_triggered": False,
        }
        return {
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "current_stage": "response_generation",
            "debug_info": _build_node_debug(state, "retrieval_evaluation", node_start),
        }

    policy_result = tool_results.get("policy_search") or {}
    policies = policy_result.get("policies", []) if isinstance(policy_result, dict) else []
    eval_result = _evaluate_policy_relevance(
        query=state.user_input,
        intent=intent,
        policy_results=policies,
    )
    tool_results["retrieval_evaluation"] = eval_result

    already_has_web = "web_search" in tool_results
    should_web_fallback = should_policy_guard and not already_has_web and eval_result.get("status") in ["poor", "borderline"]

    if should_web_fallback:
        try:
            registry = _get_tool_registry()
            web_search_input = _tool_input(
                query=state.user_input,
                max_results=3,
                search_depth="basic",
            )
            web_result = registry.invoke_tool("web_search", web_search_input)
            tool_results["web_search"] = web_result
            tool_calls.append("web_search")
            eval_result["web_fallback_triggered"] = True
        except Exception as e:
            logger.warning(f"retrieval evaluation web fallback failed: {str(e)}")
            eval_result["web_fallback_triggered"] = False
            eval_result["web_fallback_error"] = str(e)
    else:
        eval_result["web_fallback_triggered"] = False

    # 품질 저하 시 사용자 안내 메시지를 응답 생성 단계에 전달
    if (
        should_policy_guard
        and eval_result.get("status") == "poor"
        and "profile_needed" not in tool_results
        and not bool(eval_result.get("web_fallback_triggered"))
    ):
        tool_results["retrieval_quality_notice"] = {
            "status": "warning",
            "message": "질문과 정확히 맞는 정책 결과가 충분하지 않아, 조건을 더 구체적으로 주시면 정확도가 올라갑니다.",
        }

    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "current_stage": "response_generation",
        "debug_info": _build_node_debug(state, "retrieval_evaluation", node_start),
    }


# ============ 4. 응답 생성 노드 ============

def response_generation_node(state: AgentState) -> dict:
    """
    수집된 정보를 바탕으로 최종 응답 생성
    
    이 노드에서는:
    1. 도구 결과 검증
    2. OutputParser로 구조화된 응답 생성
    3. 응답 메타데이터 추가
    
    Returns:
        최종 응답
    """
    print(f"\n[NODE] response_generation_node")
    node_start = perf_counter()

    response_parts = []
    tool_sources = []
    has_policy_results = False

    # 도구 결과 기반 응답 생성
    if state.tool_results:
        if "assistant_intro" in state.tool_results:
            result = state.tool_results["assistant_intro"]
            response_parts.append(result.get("message", "안녕하세요! 청년 정책 추천을 도와드릴게요."))

        if "recommendation_explain" in state.tool_results:
            result = state.tool_results["recommendation_explain"]
            response_parts.append(result.get("message", "추천 근거 설명을 준비하지 못했습니다."))
            tool_sources.append("recommendation_explain")

        if "profile_needed" in state.tool_results:
            result = state.tool_results["profile_needed"]
            response_parts.append(result.get("message", "맞춤 추천을 위한 정보가 필요합니다."))

        if "policy_guidance" in state.tool_results:
            result = state.tool_results["policy_guidance"]
            response_parts.append(result.get("message", "정책 질문을 해주시면 맞춤으로 안내해드릴게요."))

        if "retrieval_quality_notice" in state.tool_results and "web_search" not in state.tool_results:
            result = state.tool_results["retrieval_quality_notice"]
            response_parts.append(result.get("message", "검색 결과 품질이 낮습니다."))

        # 대화 메모리 기반 응답
        if "memory_lookup" in state.tool_results:
            result = state.tool_results["memory_lookup"]
            if result.get("status") == "success":
                response_parts.append(f"🧠 대화 기억 기반 답변\n{result.get('message')}")
                tool_sources.append("memory_lookup")
            else:
                response_parts.append(result.get("message", "기억 정보를 찾지 못했습니다."))

        # 정책 검색(RAG) 결과
        if "policy_search" in state.tool_results:
            result = state.tool_results["policy_search"]
            if result.get('status') == 'success':
                policies = result.get('policies', [])
                if policies:
                    has_policy_results = True
                    synthesized_policy_answer = _synthesize_policy_answer_with_llm(state, state.tool_results)
                    if synthesized_policy_answer:
                        response_parts.append(synthesized_policy_answer)
                    else:
                        response_parts.append(f"📚 정책 검색 결과 ({len(policies)}개)")
                        response_parts.append("아래는 핵심 5개 요약입니다.")
                        for idx, p in enumerate(policies[:5], start=1):
                            title = p.get('title', '제목 없음')
                            institution = p.get('institution', '?')
                            category = p.get('category') or '-'
                            response_parts.append(f"{idx}. {title}")
                            response_parts.append(f"   - 기관: {institution}")
                            response_parts.append(f"   - 분류: {category}")
                    tool_sources.append("policy_search")
                    if len(policies) > 5:
                        response_parts.append("더 많은 결과는 '상세 결과 보기'에서 10개 단위로 확인할 수 있습니다.")
                else:
                    response_parts.append("관련 정책을 찾지 못했습니다. 조건을 조금 완화해 다시 질문해보세요.")
            elif result.get('status') == 'error':
                response_parts.append(result.get('message', '정책 검색 중 오류가 발생했습니다.'))

        # 웹 검색 결과
        if "web_search" in state.tool_results:
            result = state.tool_results["web_search"]
            if result.get('status') == 'success':
                results = result.get('results', [])
                # 1단계 방식: policy_search가 있는 경우 web_search는 보조 근거로만 사용하고 별도 문단으로 노출하지 않는다.
                if has_policy_results:
                    tool_sources.append("web_search")
                else:
                    synthesized_answer = _synthesize_web_answer_with_llm(state, result)
                    if synthesized_answer:
                        response_parts.append("🌐 웹 검색 기반 답변")
                        response_parts.append(synthesized_answer)
                        tool_sources.append("web_search")
                    elif results:
                        # LLM 합성 실패 시 폴백
                        response_parts.append(f"🔍 웹 검색 결과 ({len(results)}개):")
                        for r in results[:3]:
                            response_parts.append(f"  • {r.get('title', '제목 없음')}")
                        tool_sources.append("web_search")
            elif result.get('status') == 'warning':
                response_parts.append(result.get('message', '웹 검색 준비가 필요합니다.'))
                suggestion = result.get('suggestion')
                if suggestion:
                    response_parts.append(suggestion)
            elif result.get('status') == 'error':
                response_parts.append(result.get('message', '웹 검색 중 오류가 발생했습니다.'))
        
        # 멀티모달 파싱 결과
        if "multimodal_parser" in state.tool_results:
            result = state.tool_results["multimodal_parser"]
            if result.get('status') == 'success':
                response_parts.append(f"📄 문서 분석 완료 ({result.get('file_type')})")
                tool_sources.append("multimodal_parser")
        
        # 오류 처리
        if "error" in state.tool_results:
            error_result = state.tool_results["error"]
            response_parts.append(f"❌ {error_result.get('message', '오류 발생')}")

    # 기본 응답 생성
    if not response_parts:
        response_parts = ["요청하신 정책 정보를 검색 중입니다. 잠시만 기다려주세요."]

    final_response = "\n".join(response_parts) if response_parts else "결과를 찾을 수 없습니다."
    print(f"  ✅ Response generated ({len(final_response)} chars, {len(tool_sources)} sources)")
    response_debug = _build_node_debug(state, "response_generation", node_start)

    # 응답 메타데이터
    response_metadata = {
        "sources": tool_sources,
        "tool_calls_made": state.tool_calls,
        "context_sufficient": len(tool_sources) > 0,
        "requires_confirmation": state.privacy_consent_requested or state.deletion_confirm_requested,
        "policy_results": state.tool_results.get("policy_search", {}).get("policies", []) if state.tool_results else [],
        "policy_query": state.user_input,
        "recommendation_basis": _build_recommendation_basis(state, state.tool_results.get("_profile_used")) if state.tool_results and "policy_search" in state.tool_results else [],
        "retrieval_evaluation": state.tool_results.get("retrieval_evaluation", {}) if state.tool_results else {},
        "graph_nodes": _GRAPH_NODE_ORDER,
        "graph_path": [
            *list((response_debug.get("node_trace") or []),),
            "output",
        ],
        "graph_edges": _GRAPH_EDGES,
        "graph_path_edges": _build_path_edges([
            *list((response_debug.get("node_trace") or []),),
            "output",
            "END",
        ]),
        "node_timings_ms": response_debug.get("node_timings_ms", {}),
    }

    return {
        "final_response": final_response,
        "response_metadata": response_metadata,
        "current_stage": "response_evaluation",
        "debug_info": response_debug,
    }


def response_evaluation_node(state: AgentState) -> dict:
    """
    최종 응답 초안을 질문/대화 맥락과 대조해 평가한다.
    맥락 이탈 시 1회 재탐색 루프 또는 재질문으로 보정한다.
    """
    print(f"\n[NODE] response_evaluation_node")
    node_start = perf_counter()

    # 메모리 조회 응답은 결정론적 질의응답이므로 추가 평가 루프를 건너뛴다.
    if "memory_lookup" in (state.tool_results or {}):
        metadata = dict(state.response_metadata or {})
        metadata["response_evaluation"] = {
            "decision": "accept",
            "reason": "memory_lookup_bypass",
        }
        metadata["response_evaluation_decision"] = "accept"
        return {
            "response_metadata": metadata,
            "current_stage": "output",
            "debug_info": _build_node_debug(state, "response_evaluation", node_start),
        }

    draft_response = (state.final_response or "").strip()
    retry_count = int((state.debug_info or {}).get("response_retry_count", 0))
    eval_result = _NODE_LLM_SERVICE.evaluate_response_alignment(
        state,
        draft_response,
        state.tool_results or {},
        retry_count=retry_count,
        max_retry=1,
    )

    decision = eval_result.get("decision", "accept")
    reason = eval_result.get("reason", "")

    if decision == "clarify" and _has_actionable_policy_constraints(state.user_input):
        # 이미 실행 가능한 조건(예: 나이/지역+정책 도메인)이 있으면
        # 동일 재질문 루프 대신 1회 재탐색 또는 현재 답변 채택으로 진행한다.
        if retry_count < 1:
            decision = "retry"
            eval_result["decision"] = "retry"
            eval_result["refined_query"] = (eval_result.get("refined_query") or "").strip() or state.user_input
            eval_result["reason"] = (reason + " | clarify_suppressed_with_constraints").strip(" |")
        else:
            decision = "accept"
            eval_result["decision"] = "accept"
            eval_result["reason"] = (reason + " | clarify_suppressed_after_retry").strip(" |")

            reason = eval_result.get("reason", reason)

    print(f"  → Response eval: {decision} (reason={reason})")

    if decision == "retry":
        refined_query = (eval_result.get("refined_query") or "").strip() or state.user_input
        debug_payload = {
            "response_retry_count": retry_count + 1,
            "retry_refined_query": refined_query,
            "response_evaluation": eval_result,
        }
        return {
            "tool_calls": [],
            "tool_results": {},
            "final_response": None,
            "current_stage": "tool_calling",
            "debug_info": _build_node_debug(state, "response_evaluation", node_start, debug_payload),
        }

    if decision == "clarify":
        clarifying_question = (eval_result.get("clarifying_question") or "").strip()
        if not clarifying_question:
            clarifying_question = "질문 의도를 더 정확히 파악하고 싶어요. 원하는 지원 분야(예: 주거/취업/금융)와 지역을 알려주실래요?"

        metadata = dict(state.response_metadata or {})
        metadata["response_evaluation"] = eval_result
        metadata["response_evaluation_decision"] = "clarify"

        return {
            "final_response": clarifying_question,
            "response_metadata": metadata,
            "current_stage": "output",
            "debug_info": _build_node_debug(
                state,
                "response_evaluation",
                node_start,
                {
                    "response_retry_count": retry_count,
                    "response_evaluation": eval_result,
                },
            ),
        }

    metadata = dict(state.response_metadata or {})
    metadata["response_evaluation"] = eval_result
    metadata["response_evaluation_decision"] = "accept"

    return {
        "response_metadata": metadata,
        "current_stage": "output",
        "debug_info": _build_node_debug(
            state,
            "response_evaluation",
            node_start,
            {
                "response_retry_count": retry_count,
                "response_evaluation": eval_result,
            },
        ),
    }


# ============ 5. 조건부 분기 함수 ============

def should_check_middleware(state: AgentState) -> bool:
    """미들웨어 체크 필요 여부"""
    return state.current_stage == "init"


def should_route(state: AgentState) -> bool:
    """라우팅 필요 여부"""
    return state.current_stage == "middleware_check"


def should_call_tools(state: AgentState) -> bool:
    """도구 호출 필요 여부"""
    return state.current_stage == "tool_calling"


def should_generate_response(state: AgentState) -> bool:
    """응답 생성 필요 여부"""
    return state.current_stage in ["response_generation", "tool_calling", "retrieval_evaluation", "response_evaluation"]


def route_action(state: AgentState) -> str:
    """액션에 따른 분기"""
    if state.agent_action == "profile_matching":
        return "tool_calling"
    elif state.agent_action == "general_search":
        return "tool_calling"
    elif state.agent_action == "bookmark_management":
        return "tool_calling"
    elif state.agent_action == "data_deletion":
        return "tool_calling"
    else:
        return "response_generation"
