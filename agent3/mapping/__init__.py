"""맵핑 테이블 패키지: 추출된 원시 슬롯을 표준값으로 고정한다."""

from .slot_mapping import SlotNormalizer, normalize_region
from .category_mapping import (
    CANONICAL_FIELDS,
    map_field,
    map_field_from_many,
    infer_query_field,
)

__all__ = [
    "SlotNormalizer",
    "normalize_region",
    "CANONICAL_FIELDS",
    "map_field",
    "map_field_from_many",
    "infer_query_field",
]
