"""agent2 FastAPI 서버.

라우트
- GET  /            : 새 채팅 UI
- POST /api/chat    : 파이프라인 처리 (plan/tools/timings 포함 응답)
- GET  /api/models  : 사용 가능 모델
- GET  /api/health  : 헬스체크
- GET  /api/graph   : 파이프라인 노드/엣지 정적 정의(UI 그래프용)
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent3.config import settings
from agent3.service import AgentService

app = FastAPI(title="청년정책 에이전트 v3 (LangGraph)", version="3.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

service = AgentService()

# LangGraph StateGraph 노드/엣지 (UI 그래프 렌더링용 정적 정의)
GRAPH_NODES = [
    "security", "context_load", "planner", "memory",
    "slots", "tools", "responder", "judge", "retry_prep",
    "finalize_blocked", "finalize_memory", "finalize_answer",
]
GRAPH_EDGES = [
    {"from": "security", "to": "finalize_blocked", "label": "차단"},
    {"from": "security", "to": "context_load", "label": "통과"},
    {"from": "context_load", "to": "planner"},
    {"from": "planner", "to": "memory"},
    {"from": "memory", "to": "finalize_memory", "label": "메모리 처리"},
    {"from": "memory", "to": "slots", "label": "통과"},
    {"from": "slots", "to": "tools"},
    {"from": "tools", "to": "responder"},
    {"from": "responder", "to": "judge"},
    {"from": "judge", "to": "retry_prep", "label": "retry"},
    {"from": "judge", "to": "finalize_answer", "label": "accept/clarify"},
    {"from": "retry_prep", "to": "tools"},
    {"from": "finalize_blocked", "to": "output"},
    {"from": "finalize_memory", "to": "output"},
    {"from": "finalize_answer", "to": "output"},
]


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    query: str = Field(..., description="사용자 질문")
    user_code: Optional[str] = Field(None, description="유저코드(장기기억 식별자)")
    provider: Optional[str] = Field(None, description="LLM provider")
    model_name: Optional[str] = Field(None, description="LLM 모델명")
    file_path: Optional[str] = Field(None, description="업로드된 파일 경로")
    file_type: Optional[str] = Field(None, description="파일 타입")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"), media_type="text/html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.0.0"}


@app.get("/api/graph")
def graph() -> dict[str, Any]:
    return {"nodes": GRAPH_NODES, "edges": GRAPH_EDGES}


@app.get("/api/models")
def models() -> dict[str, Any]:
    openai_ok = bool(settings.openai_api_key) and importlib.util.find_spec("langchain_openai") is not None
    google_ok = bool(settings.google_api_key) and importlib.util.find_spec("langchain_google_genai") is not None
    anthropic_ok = bool(settings.anthropic_api_key) and importlib.util.find_spec("langchain_anthropic") is not None

    catalog = [
        {"provider": "openai", "model": "gpt-4o-mini", "label": "OpenAI GPT-4o Mini", "available": openai_ok},
        {"provider": "openai", "model": "gpt-4o", "label": "OpenAI GPT-4o", "available": openai_ok},
        {"provider": "openai", "model": "gpt-4.1-mini", "label": "OpenAI GPT-4.1 Mini", "available": openai_ok},
        {"provider": "openai", "model": "gpt-4.1", "label": "OpenAI GPT-4.1", "available": openai_ok},
        {"provider": "openai", "model": "gpt-4-turbo", "label": "OpenAI GPT-4 Turbo", "available": openai_ok},
        {"provider": "openai", "model": "o3-mini", "label": "OpenAI o3-mini (추론)", "available": openai_ok},
        {"provider": "openai", "model": "o1", "label": "OpenAI o1 (상위 추론)", "available": openai_ok},
        {"provider": "gemini", "model": "gemini-2.5-flash", "label": "Google Gemini 2.5 Flash", "available": google_ok},
        {"provider": "anthropic", "model": "claude-3-5-sonnet-latest", "label": "Claude 3.5 Sonnet", "available": anthropic_ok},
    ]
    return {"models": catalog, "count": len(catalog)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        result = service.process(
            user_id=req.user_id,
            query=req.query,
            user_code=req.user_code,
            provider=req.provider,
            model_name=req.model_name,
            file_path=req.file_path,
            file_type=req.file_type,
        )
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    def event_source():
        try:
            for ev in service.process_stream(
                user_id=req.user_id,
                query=req.query,
                user_code=req.user_code,
                provider=req.provider,
                model_name=req.model_name,
                file_path=req.file_path,
                file_type=req.file_type,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        ext = Path(file.filename or "").suffix.lower()
        safe = f"{uuid.uuid4().hex}{ext}"
        dest = UPLOAD_DIR / safe
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        ftype = (
            "pdf" if ext == ".pdf"
            else "image" if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            else "text"
        )
        return {"status": "success", "file_path": str(dest), "file_name": file.filename, "file_type": ftype}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
