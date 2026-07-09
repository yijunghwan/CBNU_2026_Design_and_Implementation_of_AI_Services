"""인덱스 빌더: policy_docs.csv / content_docs.csv → Chroma 벡터스토어 2개.

- 정책/컨텐츠를 완전히 분리된 컬렉션으로 저장한다.
- 긴 본문은 청크로 분할하되, 하드조건 메타데이터를 모든 청크에 부착한다.
- doc_id를 메타데이터에 보존해 검색 단계에서 문서 단위로 dedup 가능하게 한다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from agent3.config import settings

# 정책 메타데이터 컬럼(하드조건 평가에 사용) — 없으면 빈 값
_POLICY_META_COLS = [
    "doc_id", "title", "url", "category_raw",
    "field", "region", "age_min", "age_max", "income_max", "marriage",
]
_CONTENT_META_COLS = [
    "doc_id", "title", "url", "category_raw", "has_attachment", "attachment_type",
]


def _clean_meta_value(value):
    """Chroma 메타데이터는 None을 허용하지 않으므로 표준화한다."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _rows_to_documents(df: pd.DataFrame, meta_cols: list[str]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        base_meta = {col: _clean_meta_value(row.get(col)) for col in meta_cols}
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            documents.append(Document(page_content=chunk, metadata=meta))
    return documents


def _build_store(df: pd.DataFrame, meta_cols: list[str], persist_dir: Path, label: str) -> int:
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = _rows_to_documents(df, meta_cols)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    print(f"[{label}] 문서 {len(df)}건 → 청크 {len(docs)}개 임베딩 중...")
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    store.persist()
    print(f"[{label}] 완료 → {persist_dir}")
    return len(docs)


def build_policy_index() -> int:
    df = pd.read_csv(settings.policy_docs_path, dtype=str, encoding="utf-8-sig")
    return _build_store(df, _POLICY_META_COLS, settings.policy_store_dir, "policy")


def build_content_index() -> int:
    df = pd.read_csv(settings.content_docs_path, dtype=str, encoding="utf-8-sig")
    return _build_store(df, _CONTENT_META_COLS, settings.content_store_dir, "content")


def main() -> int:
    if not settings.policy_docs_path.exists() or not settings.content_docs_path.exists():
        print("[ERROR] docs CSV가 없습니다. 먼저 build_rag_docs를 실행하세요.")
        return 1
    p = build_policy_index()
    c = build_content_index()
    print(f"[ALL DONE] policy chunks={p}, content chunks={c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
