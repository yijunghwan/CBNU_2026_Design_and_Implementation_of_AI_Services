"""agent3 설정: .env 기반, 경로/모델/키 관리 (단독 실행, 자체 완결)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
AGENT3_DIR = ROOT / "agent3"
DATA_DIR = AGENT3_DIR / "data"


@dataclass
class Settings:
    # 경로
    policy_docs_path: Path = DATA_DIR / "policy_docs.csv"
    content_docs_path: Path = DATA_DIR / "content_docs.csv"
    policy_store_dir: Path = DATA_DIR / "vectorstore_policy"
    content_store_dir: Path = DATA_DIR / "vectorstore_content"

    # 임베딩
    embedding_model: str = os.getenv("AGENT2_EMBEDDING_MODEL", "text-embedding-3-small")

    # 청크
    chunk_size: int = int(os.getenv("AGENT2_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("AGENT2_CHUNK_OVERLAP", "150"))

    # 기본 LLM (사용자 선택으로 오버라이드)
    default_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    default_model: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    # 키
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: str | None = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))


settings = Settings()
