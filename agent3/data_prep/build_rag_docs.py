"""agent2 전용 RAG 문서 생성 (정책/컨텐츠 분리).

입력(기존 처리본 재사용)
  data/processed/youth_policy_processed.csv
  data/processed/youth_content_processed.csv

출력
  agent2/data/policy_docs.csv   (구조화 하드조건 메타데이터 포함)
  agent2/data/content_docs.csv  (본문 의미 중심)

원칙
- 슬롯 값은 맵핑 테이블로 표준값 고정 (지역=표준지역, 나이=정수, 분야=표준6종).
- 메타데이터가 없으면 빈 값으로 두고, 평가 단계에서 '없으면 평가 제외'한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from agent3.mapping import map_field_from_many, normalize_region

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "agent2" / "data"

POLICY_IN = PROCESSED / "youth_policy_processed.csv"
CONTENT_IN = PROCESSED / "youth_content_processed.csv"
POLICY_OUT = OUT_DIR / "policy_docs.csv"
CONTENT_OUT = OUT_DIR / "content_docs.csv"

# mrgSttsCd 표준: 55001=기혼, 55002=미혼, 55003=제한없음(평가 제외)
_MARRIAGE_CODE = {
    "55001": "married",
    "55002": "single",
    "55003": None,
}


def _to_int(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        num = float(text)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return int(num)


def _marriage(value) -> Optional[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace(".0", "")
    return _MARRIAGE_CODE.get(text)


def build_policy_docs() -> pd.DataFrame:
    df = pd.read_csv(POLICY_IN, dtype=str, encoding="utf-8-sig")

    region = (
        df.get("operInstCdNm")
        .fillna(df.get("sprvsnInstCdNm", ""))
        .fillna("")
        .map(lambda v: normalize_region(v) or "")
    )
    field = [
        map_field_from_many(m, l) or ""
        for m, l in zip(df.get("mclsfNm", "").fillna(""), df.get("lclsfNm", "").fillna(""))
    ]

    title = df.get("plcyNm", "").fillna("").astype(str)
    body = df.get("policy_text", "").fillna("").astype(str)
    text = (title + "\n" + body).str.strip()

    out = pd.DataFrame(
        {
            "doc_id": "policy:" + df.get("plcyNo", "").fillna("").astype(str),
            "title": title,
            "text": text,
            "url": df.get("aplyUrlAddr", "").fillna("").astype(str),
            "category_raw": df.get("mclsfNm", "").fillna("").astype(str),
            "field": field,
            "region": region,
            "age_min": [_to_int(v) for v in df.get("sprtTrgtMinAge", "")],
            "age_max": [_to_int(v) for v in df.get("sprtTrgtMaxAge", "")],
            "income_max": [_to_int(v) for v in df.get("earnMaxAmt", "")],
            "marriage": [_marriage(v) for v in df.get("mrgSttsCd", "")],
        }
    )
    out = out[out["text"].str.len() > 0].drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    return out


def build_content_docs() -> pd.DataFrame:
    df = pd.read_csv(CONTENT_IN, dtype=str, encoding="utf-8-sig")

    title = df.get("pstTtl", "").fillna("").astype(str)
    body = df.get("content_text", "").fillna("").astype(str)
    text = (title + "\n" + body).str.strip()

    out = pd.DataFrame(
        {
            "doc_id": "content:" + df.get("pstSn", "").fillna("").astype(str),
            "title": title,
            "text": text,
            "url": df.get("pstUrlAddr", "").fillna("").astype(str),
            "category_raw": df.get("pstSeNm", "").fillna("").astype(str),
            "has_attachment": df.get("has_attachment", "").fillna("").astype(str),
            "attachment_type": df.get("attachment_type", "").fillna("").astype(str),
        }
    )
    out = out[out["text"].str.len() > 0].drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    return out


def main() -> int:
    if not POLICY_IN.exists() or not CONTENT_IN.exists():
        print("[ERROR] 처리본 CSV가 없습니다. 먼저 전처리를 실행하세요.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = build_policy_docs()
    content = build_content_docs()

    policy.to_csv(POLICY_OUT, index=False, encoding="utf-8-sig")
    content.to_csv(CONTENT_OUT, index=False, encoding="utf-8-sig")

    print(f"[DONE] policy_docs rows={len(policy)} -> {POLICY_OUT}")
    print(f"       field 분포:\n{policy['field'].value_counts(dropna=False).to_string()}")
    print(f"       region 채움={int((policy['region'] != '').sum())}/{len(policy)}")
    print(f"       age_min 채움={int(policy['age_min'].notna().sum())}, income_max 채움={int(policy['income_max'].notna().sum())}")
    print(f"[DONE] content_docs rows={len(content)} -> {CONTENT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
