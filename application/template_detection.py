"""Evaluation-only AS/400 template detection; never used for extraction selection."""

import re
import unicodedata
from dataclasses import dataclass

from application.extractor_registry import TEMPLATE_DETECTORS

MIN_SIGNALS = 2
MIN_FIXED_LABELS = 2


@dataclass(frozen=True)
class TemplateDetection:
    template: str
    detector_id: str
    detector_version: str
    signals: dict[str, tuple[str, ...]]
    reason: str | None


def _normalize(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip().upper()


def _headers(pages):
    found = set()
    for page in pages:
        for table in page.tables:
            for row in table.rows:
                cells = tuple(_normalize(cell) for cell in row if _normalize(cell))
                if cells:
                    found.add(cells)
    return found


def _geometry(pages, kind):
    for page in pages:
        for table in page.tables:
            x0 = table.bounds.x0 / page.width
            top = table.bounds.top / page.height
            x1 = table.bounds.x1 / page.width
            bottom = table.bounds.bottom / page.height
            wide = x0 <= 0.08 and x1 >= 0.90
            if kind == "split-detail-band" and wide and 0.35 <= top <= 0.39 and 0.70 <= bottom <= 0.77:
                return True
            if kind == "single-detail-band" and wide and 0.31 <= top <= 0.35 and 0.65 <= bottom <= 0.70:
                return True
    return False


def detect_template(pages, detectors=TEMPLATE_DETECTORS):
    pages = tuple(pages)
    if len(pages) != 1 or not pages[0].width or not pages[0].height:
        return TemplateDetection("unknown", "as400-template-router", "1", {}, "unsupported-page-count")
    ratio = min(pages[0].width, pages[0].height) / max(pages[0].width, pages[0].height)
    if not 0.68 <= ratio <= 0.73 or pages[0].width > pages[0].height:
        return TemplateDetection("unknown", "as400-template-router", "1", {}, "incompatible-page-format")

    headings = _headers(pages)
    text = _normalize("\n".join(page.text for page in pages))
    matches = []
    observed = {}
    for detector in detectors:
        header = tuple(_normalize(item) for item in detector.headers) in headings
        labels = tuple(label for label in detector.labels if _normalize(label) in text)
        geometry = _geometry(pages, detector.geometry)
        signals = {"column_header": ("matched",) if header else (),
                   "fixed_labels": labels, "geometry": (detector.geometry,) if geometry else ()}
        observed[detector.id] = signals
        # Header order is mandatory and needs one independent label/geometry signal.
        signal_count = int(header) + int(len(labels) >= MIN_FIXED_LABELS) + int(geometry)
        if header and signal_count >= MIN_SIGNALS:
            matches.append((detector, signals))

    if len(matches) == 1:
        detector, _ = matches[0]
        return TemplateDetection(detector.id.removeprefix("as400-").upper(),
                                 detector.id, detector.version, observed, None)
    if len(matches) > 1:
        return TemplateDetection("unknown", "as400-template-router", "1", observed, "ambiguous-template-match")
    return TemplateDetection("unknown", "as400-template-router", "1", observed, "insufficient-structural-signals")
