"""agent3 개발 서버 실행: python -m agent3.run_server"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("청년정책 에이전트 v3 서버 (LangGraph)")
    print("  - UI:  http://localhost:8002")
    print("  - API: http://localhost:8002/docs")
    print("=" * 70)
    uvicorn.run("agent3.api.app:app", host="0.0.0.0", port=8002, reload=False, log_level="info")
