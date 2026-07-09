"""LLM 클라이언트: provider별 모델 생성 + 안전한 호출/ JSON 파싱.

- 사용자가 고른 provider/model로 응답 LLM을 만든다.
- 플래너/판정 같은 보조 LLM은 작은 기본 모델을 쓸 수 있다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser

from agent3.config import settings

logger = logging.getLogger(__name__)


def create_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1200,
):
    """provider에 맞는 LangChain Chat 모델 생성."""
    p = (provider or settings.default_provider or "openai").lower()
    model = model_name or settings.default_model

    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if p in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    if p == "ollama":
        from langchain_community.llms import Ollama

        return Ollama(model=model, temperature=temperature)

    raise ValueError(f"지원하지 않는 provider: {p}")


def invoke_text(llm, prompt: str) -> str:
    """LLM 호출 후 텍스트만 반환. 실패 시 빈 문자열."""
    try:
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM invoke_text 실패: %s", e)
        return ""


def _strip_code_fence(text: str) -> str:
    return (text or "").replace("```json", "").replace("```", "").strip()


# LangChain OutputParser (구조화 출력) — 1차 파서로 사용
_json_output_parser = JsonOutputParser()


def invoke_json(llm, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """LLM 호출 후 구조화 JSON 파싱.

    1차: LangChain ``JsonOutputParser``로 파싱(구조화 출력 컴포넌트).
    2차: 실패 시 코드펜스 제거 후 ``json.loads`` 보정.
    3차: 그래도 실패하면 fallback 반환.
    """
    raw = invoke_text(llm, prompt)
    if not raw:
        return dict(fallback)

    # 1차: LangChain OutputParser
    try:
        parsed = _json_output_parser.parse(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:  # noqa: BLE001
        logger.debug("JsonOutputParser 실패, 보정 파싱 시도: %s", e)

    # 2차: 코드펜스 제거 후 표준 json 파싱
    try:
        parsed = json.loads(_strip_code_fence(raw))
        if isinstance(parsed, dict):
            return parsed
        return dict(fallback)
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug("JSON 파싱 실패(fallback 사용): %s | raw=%s", e, raw[:200])
        return dict(fallback)
