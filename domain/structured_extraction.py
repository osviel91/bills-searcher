from datetime import date
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class TextBlock:
    id: str
    text: str
    bounds: BoundingBox


@dataclass(frozen=True)
class PDFToken:
    id: str
    text: str
    bounds: BoundingBox
    character_ids: tuple[str, ...]
    character_bounds: tuple[BoundingBox, ...]


@dataclass(frozen=True)
class TableRegion:
    id: str
    bounds: BoundingBox
    rows: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class PDFPage:
    number: int
    width: float
    height: float
    text: str
    blocks: tuple[TextBlock, ...] = ()
    tables: tuple[TableRegion, ...] = ()


@dataclass(frozen=True)
class Evidence:
    page_number: int
    quote: str
    bounds: BoundingBox | None = None
    block_id: str | None = None
    label_quote: str | None = None
    label_bounds: BoundingBox | None = None
    method: str | None = None
    extractor_id: str | None = None
    extractor_version: str | None = None
    source_token_ids: tuple[str, ...] = ()


FieldValue = str | Decimal | date | int | bool | tuple[Decimal, ...] | None
ExtractionState = Literal[
    "confirmed", "absent", "ambiguous", "not_applicable", "unreadable",
    "conflict", "insufficient_evidence", "unverified_semantics",
]


@dataclass(frozen=True)
class ExtractedField:
    name: str
    value: FieldValue = None
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()
    status: ExtractionState | None = None
    candidates: tuple[FieldValue, ...] = ()


@dataclass(frozen=True)
class ProposedLine:
    description: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractionExecution:
    id: str
    document_id: str
    extractor_version: str
    status: Literal["succeeded", "failed"]
    pages: tuple[PDFPage, ...] = ()
    fields: tuple[ExtractedField, ...] = ()
    lines: tuple[ProposedLine, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
