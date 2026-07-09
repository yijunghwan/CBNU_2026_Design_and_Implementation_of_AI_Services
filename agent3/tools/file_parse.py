"""파일 파싱 툴: PDF / 이미지(OCR) / 텍스트에서 본문 추출.

의존성이 없으면 안내 메시지와 함께 안전하게 실패한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CHARS = 4000


def _parse_pdf(path: str) -> dict[str, Any]:
    try:
        import PyPDF2

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join((page.extract_text() or "") for page in reader.pages)
        return {"status": "success", "file_type": "pdf", "pages": len(reader.pages), "text": text[:_MAX_CHARS]}
    except ImportError:
        return {"status": "error", "message": "PyPDF2 미설치 (pip install PyPDF2)", "text": ""}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"PDF 파싱 실패: {e}", "text": ""}


def _parse_image(path: str) -> dict[str, Any]:
    try:
        from PIL import Image
        import pytesseract

        text = pytesseract.image_to_string(Image.open(path), lang="kor+eng")
        return {"status": "success", "file_type": "image", "text": text[:_MAX_CHARS]}
    except ImportError:
        return {"status": "error", "message": "Pillow/pytesseract 미설치", "text": ""}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"이미지 OCR 실패: {e}", "text": ""}


def _parse_text(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {"status": "success", "file_type": "text", "text": text[:_MAX_CHARS]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"텍스트 파싱 실패: {e}", "text": ""}


def parse_file(file_path: str, file_type: str | None = None) -> dict[str, Any]:
    if not file_path or not Path(file_path).exists():
        return {"status": "error", "message": "파일을 찾을 수 없어요.", "text": ""}

    ext = Path(file_path).suffix.lower()
    ftype = file_type or (
        "pdf" if ext == ".pdf"
        else "image" if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        else "text"
    )
    if ftype == "pdf":
        return _parse_pdf(file_path)
    if ftype == "image":
        return _parse_image(file_path)
    return _parse_text(file_path)
