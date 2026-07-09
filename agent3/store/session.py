"""세션(단기) 저장소: 프로세스 메모리, user_id별 최근 대화.

- 최근 max_items 메시지 보관, 컨텍스트에는 최근 N턴만 주입(오케스트레이터/빌더에서 조절).
"""

from __future__ import annotations

from typing import Dict, List


class SessionStore:
    def __init__(self, max_items: int = 60):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._max_items = max_items

    def get_recent(self, user_id: str, limit: int = 20) -> List[Dict[str, str]]:
        return self._store.get(user_id, [])[-limit:]

    def append(self, user_id: str, role: str, content: str) -> None:
        self._store.setdefault(user_id, []).append({"role": role, "content": content})
        if len(self._store[user_id]) > self._max_items:
            self._store[user_id] = self._store[user_id][-self._max_items:]

    def clear(self, user_id: str) -> None:
        self._store.pop(user_id, None)
