"""웹검색 툴: Tavily 기반 보완 검색 (최신정보/정책용어).

키가 없으면 status=warning으로 안전하게 빈 결과를 반환한다.
"""

from __future__ import annotations

import logging
from typing import Any

from agent3.config import settings

logger = logging.getLogger(__name__)


def web_search(query: str, *, max_results: int = 3, search_depth: str = "basic") -> dict[str, Any]:
    if not settings.tavily_api_key:
        return {
            "status": "warning",
            "message": "웹검색 키(TAVILY_API_KEY)가 없어 웹검색을 건너뛰었어요.",
            "results": [],
        }

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
        )
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            }
            for r in response.get("results", [])
        ]
        return {
            "status": "success",
            "query": query,
            "answer": response.get("answer", ""),
            "results": results,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search 실패: %s", e)
        return {"status": "error", "message": f"웹검색 실패: {e}", "results": []}
