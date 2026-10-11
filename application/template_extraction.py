"""Offline, evidence-backed header and summary extraction for AS/400 layouts."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from application.extractor_registry import template_extractor_for
from application.template_detection import TemplateDetection, detect_template
from domain.structured_extraction import BoundingBox, Evidence, ExtractedField, PDFPage, PDFToken, ProposedLine
from infrastructure.pdf_extraction import extract_pdf, extract_pdf_tokens


@dataclass(frozen=True)
class Span:
    token: PDFToken
    start: int
    end: int
    text: str
    bounds: BoundingBox

    @property
    def token_ids(self):
        return tuple(self.token.character_ids[self.start:self.end])


@dataclass(frozen=True)
class TemplateExtraction:
    classification: TemplateDetection
    extractor_id: str | None
    extractor_version: str | None
    fields: tuple[ExtractedField, ...]
    warnings: tuple[str, ...]


def _norm(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def _span(token, start, end):
    chars = token.character_bounds[start:end]
    return Span(token, start, end, token.text[start:end], BoundingBox(
        min(char.x0 for char in chars), min(char.top for char in chars),
        max(char.x1 for char in chars), max(char.bottom for char in chars),
    ))


def _matches(tokens, pattern):
    result = []
    regex = re.compile(pattern, re.IGNORECASE)
    for token in tokens:
        if len(token.text) != len(token.character_bounds):
            continue
        result.extend(_span(token, match.start(), match.end()) for match in regex.finditer(token.text))
    return result


def _full_matches(tokens, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    result = []
    for token in tokens:
        if len(token.text) == len(token.character_bounds):
            match = regex.fullmatch(token.text)
            if match:
                result.append(_span(token, match.start(), match.end()))
    return result


def _rows(tokens, page, tolerance=2.5):
    rows = []
    for token in sorted((t for t in tokens if "-v" not in t.id), key=lambda t: (t.bounds.top, t.bounds.x0)):
        if rows and abs(token.bounds.top - rows[-1][0]) <= tolerance:
            rows[-1][1].append(token)
        else:
            rows.append((token.bounds.top, [token]))
    return [sorted(row, key=lambda t: t.bounds.x0) for _, row in rows]


def _labels(tokens, page, phrases, zone):
    hits = []
    for row in _rows(tokens, page):
        for length in range(1, min(5, len(row)) + 1):
            for offset in range(len(row) - length + 1):
                group = row[offset:offset + length]
                bounds = BoundingBox(min(t.bounds.x0 for t in group), min(t.bounds.top for t in group),
                                     max(t.bounds.x1 for t in group), max(t.bounds.bottom for t in group))
                center_y = (bounds.top + bounds.bottom) / 2 / page.height
                center_x = (bounds.x0 + bounds.x1) / 2 / page.width
                if not zone[0] <= center_x <= zone[2] or not zone[1] <= center_y <= zone[3]:
                    continue
                value = " ".join(_norm(t.text) for t in group)
                if value in phrases:
                    hits.append((value, tuple(_span(t, 0, len(t.text)) for t in group), bounds))
    # Some narrow PDF table headers are emitted as individual glyph tokens.
    letter_groups = []
    for row in _rows(tokens, page):
        letters = [_norm(token.text) for token in row]
        for target in ("TIPO", "IVA"):
            for offset in range(len(row) - len(target) + 1):
                group = row[offset:offset + len(target)]
                parts = letters[offset:offset + len(target)]
                if "".join(parts) != target or any(len(part) != 1 for part in parts):
                    continue
                spans = tuple(_span(token, 0, len(token.text)) for token in group)
                bounds = BoundingBox(min(span.bounds.x0 for span in spans), min(span.bounds.top for span in spans),
                                     max(span.bounds.x1 for span in spans), max(span.bounds.bottom for span in spans))
                x, y = (bounds.x0 + bounds.x1) / 2 / page.width, (bounds.top + bounds.bottom) / 2 / page.height
                if zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]:
                    letter_groups.append((target, spans, bounds))
    for tipo in (hit for hit in letter_groups if hit[0] == "TIPO"):
        for iva in (hit for hit in letter_groups if hit[0] == "IVA"):
            tx, ix = (tipo[2].x0 + tipo[2].x1) / 2 / page.width, (iva[2].x0 + iva[2].x1) / 2 / page.width
            ty, iy = (tipo[2].top + tipo[2].bottom) / 2 / page.height, (iva[2].top + iva[2].bottom) / 2 / page.height
            if abs(tx - ix) <= .10 and abs(ty - iy) <= .03:
                bounds = BoundingBox(min(tipo[2].x0, iva[2].x0), min(tipo[2].top, iva[2].top),
                                     max(tipo[2].x1, iva[2].x1), max(tipo[2].bottom, iva[2].bottom))
                hits.append(("TIPO IVA", tipo[1] + iva[1], bounds))
    if "TIPO IVA" in phrases and not any(hit[0] == "TIPO IVA" for hit in hits):
        tipos = [span for span in _matches(tokens, r"TIPO")
                 if zone[0] <= (span.bounds.x0 + span.bounds.x1) / 2 / page.width <= zone[2]
                 and zone[1] <= (span.bounds.top + span.bounds.bottom) / 2 / page.height <= zone[3]]
        ivas = [span for span in _matches(tokens, r"IVA")
                if zone[0] <= (span.bounds.x0 + span.bounds.x1) / 2 / page.width <= zone[2]
                and zone[1] <= (span.bounds.top + span.bounds.bottom) / 2 / page.height <= zone[3]]
        for tipo in tipos:
            for iva in ivas:
                tx = (tipo.bounds.x0 + tipo.bounds.x1) / 2 / page.width
                ix = (iva.bounds.x0 + iva.bounds.x1) / 2 / page.width
                ty = (tipo.bounds.top + tipo.bounds.bottom) / 2 / page.height
                iy = (iva.bounds.top + iva.bounds.bottom) / 2 / page.height
                if abs(tx - ix) <= .10 and 0 < iy - ty <= .025:
                    bounds = BoundingBox(min(tipo.bounds.x0, iva.bounds.x0), min(tipo.bounds.top, iva.bounds.top),
                                         max(tipo.bounds.x1, iva.bounds.x1), max(tipo.bounds.bottom, iva.bounds.bottom))
                    hits.append(("TIPO IVA", (tipo, iva), bounds))
    # A repeated label remains visible to the caller so it can abstain.
    unique = {}
    for hit in hits:
        unique[(hit[0], hit[2])] = hit
    return list(unique.values())


def _evidence(value: Span, label, method, extractor):
    label_text = " ".join(span.text for span in label[1]) if label else None
    label_bounds = label[2] if label else None
    return Evidence(
        page_number=1, quote=value.text, bounds=value.bounds, block_id=value.token.id,
        label_quote=label_text, label_bounds=label_bounds, method=method,
        extractor_id=extractor.id, extractor_version=extractor.version,
        source_token_ids=value.token_ids,
    )


def _field(name, status, *, value=None, evidence=(), warning=None, candidates=()):
    return ExtractedField(name, value, tuple(evidence), (warning,) if warning else (), status, tuple(candidates))


def _number(text):
    raw = text.strip().replace("€", "").replace("EUR", "").replace("%", "").replace(" ", "")
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _associate(label, values, page, *, zone, x_limits=None):
    """Associate one visual cell by its column interval and vertical position."""
    _, _, label_box = label
    lx = (label_box.x0 + label_box.x1) / 2 / page.width
    ly = (label_box.top + label_box.bottom) / 2 / page.height
    ordered = sorted(values, key=lambda hit: abs((hit.bounds.x0 + hit.bounds.x1) / 2 / page.width - lx))
    if x_limits:
        ordered = [hit for hit in ordered if x_limits[0] <= (hit.bounds.x0 + hit.bounds.x1) / 2 / page.width <= x_limits[1]]
    ordered = [hit for hit in ordered
               if max(zone[1], label_box.bottom / page.height) <=
               (hit.bounds.top + hit.bounds.bottom) / 2 / page.height <= min(zone[3], ly + 0.075)]
    return ordered


def _cell_limits(label, labels, page, page_left=.04, page_right=.98):
    centers = sorted(set(round((hit[2].x0 + hit[2].x1) / 2 / page.width, 5) for hit in labels))
    center = round((label[2].x0 + label[2].x1) / 2 / page.width, 5)
    index = centers.index(center)
    left = page_left if index == 0 else (centers[index - 1] + center) / 2
    right = page_right if index == len(centers) - 1 else (center + centers[index + 1]) / 2
    return left - .015, right + .015


def _header_field(name, pattern, value_pattern, tokens, page, extractor, header_zone, parser=str):
    labels = _labels(tokens, page, {pattern}, header_zone)
    if not labels:
        return _field(name, "absent", warning="label-not-present")
    if len(labels) != 1:
        return _field(name, "ambiguous", warning="duplicate-header-label")
    label = labels[0]
    label_center = (label[2].x0 + label[2].x1) / 2 / page.width
    candidates = _full_matches(tokens, value_pattern)
    left, right = (label_center - 0.10, label_center + 0.10)
    candidates = [candidate for candidate in candidates
                  if left <= (candidate.bounds.x0 + candidate.bounds.x1) / 2 / page.width <= right
                  and header_zone[1] <= (candidate.bounds.top + candidate.bounds.bottom) / 2 / page.height <= header_zone[3]]
    candidates.sort(key=lambda item: abs((item.bounds.x0 + item.bounds.x1) / 2 / page.width - label_center))
    if not candidates:
        return _field(name, "unreadable", warning="label-present-value-not-parseable-in-header-cell")
    if len(candidates) > 1 and abs((candidates[0].bounds.x0 + candidates[0].bounds.x1) / 2 / page.width -
                                  (candidates[1].bounds.x0 + candidates[1].bounds.x1) / 2 / page.width) < 0.002:
        return _field(name, "ambiguous", warning="multiple-header-values-in-cell")
    hit = candidates[0]
    value = parser(hit.text)
    if value is None:
        return _field(name, "unreadable", evidence=(_evidence(hit, label, "header-label-same-column", extractor),),
                      warning="header-value-parse-failed")
    return _field(name, "confirmed", value=value,
                  evidence=(_evidence(hit, label, "header-label-same-column", extractor),))


def _summary_fields(pages, token_pages, extractor, template):
    fields = []
    summary_zone = (0.04, 0.69, 0.98, 0.92) if template == "T02" else (0.04, 0.77, 0.98, 0.92)
    page = pages[0]
    tokens = token_pages[0]
    phrases = {"BASE IMPONIBLE", "CUOTA IVA", "TOTAL FACTURA", "TOTAL A PAGAR", "IVA", "TOTAL EUROS",
               "COMISION", "FORFAIT GESTION", "GASTOS ENVIO"}
    all_labels = _labels(tokens, page, phrases, summary_zone)
    if template == "T02":
        cuota = [hit for hit in all_labels if hit[0] == "CUOTA IVA"]
        all_labels = [hit for hit in all_labels if hit[0] != "IVA" or not any(
            hit[2].top == label[2].top and label[2].x0 <= hit[2].x0 <= label[2].x1 for label in cuota)]

    def amount_field(name, phrase, *, x_band=None):
        labels = [hit for hit in all_labels if hit[0] == phrase]
        if not labels:
            return _field(name, "absent", warning="summary-label-not-present")
        if len(labels) > 1:
            return _field(name, "ambiguous", warning="duplicate-summary-label")
        label = labels[0]
        spans = _matches(tokens, r"-?\d{1,3}(?:\.\d{3})*(?:,\d{2})|-?\d+(?:,\d{2})?")
        spans = [hit for hit in spans if _number(hit.text) is not None]
        values = _associate(label, spans, page, zone=summary_zone,
                            x_limits=x_band or _cell_limits(label, all_labels, page))
        # Rates and bases can be adjacent within T01's merged IVA cell; the exact column distinguishes them.
        values = [hit for hit in values if "%" not in hit.text]
        if not values:
            return _field(name, "insufficient_evidence", warning="no-amount-in-labeled-summary-cell")
        nearest = abs((values[0].bounds.x0 + values[0].bounds.x1) / 2 / page.width -
                      (label[2].x0 + label[2].x1) / 2 / page.width)
        ties = [hit for hit in values if abs(abs((hit.bounds.x0 + hit.bounds.x1) / 2 / page.width -
                                                  (label[2].x0 + label[2].x1) / 2 / page.width) - nearest) < 0.002]
        if len(ties) > 1:
            return _field(name, "conflict", warning="multiple-summary-values-in-column")
        return _field(name, "confirmed", value=_number(values[0].text),
                      evidence=(_evidence(values[0], label, "summary-label-column", extractor),))

    if template == "T02":
        for name, label in (("tax_base", "BASE IMPONIBLE"), ("tax_amount", "CUOTA IVA"),
                            ("total_invoice", "TOTAL FACTURA"), ("total_due", "TOTAL A PAGAR")):
            fields.append(amount_field(name, label))
        # T02 exposes the VAT rate in its detail column; collect distinct rates, never assume one.
        vat_header = _labels(tokens, page, {"TIPO IVA"}, (0.68, 0.31, 0.84, 0.38))
        rate_tokens = [hit for hit in _matches(tokens, r"\d{1,3}(?:[.,]\d{1,2})?%?")
                       if 0.70 <= (hit.bounds.x0 + hit.bounds.x1) / 2 / page.width <= 0.78
                       and 0.35 <= (hit.bounds.top + hit.bounds.bottom) / 2 / page.height <= 0.69]
        rates = {}
        for hit in rate_tokens:
            value = _number(hit.text)
            if value is not None:
                rates.setdefault(value, []).append(hit)
        if rates:
            evidence = tuple(_evidence(hit, vat_header[0] if len(vat_header) == 1 else None,
                                       "detail-vat-column", extractor)
                             for occurrences in rates.values() for hit in occurrences)
            fields.append(_field("tax_rates", "confirmed", value=tuple(sorted(rates)), evidence=evidence))
        elif vat_header:
            fields.append(_field("tax_rates", "not_applicable", warning="vat-column-present-but-blank"))
        else:
            fields.append(_field("tax_rates", "insufficient_evidence", warning="vat-rate-column-not-located"))
        return tuple(fields)

    base = amount_field("tax_base", "BASE IMPONIBLE", x_band=(0.43, 0.73))
    fields.append(base)
    iva_labels = [hit for hit in all_labels if hit[0] == "IVA" and .60 <= (hit[2].x0 + hit[2].x1) / 2 / page.width <= .88]
    if not iva_labels:
        fields.append(_field("tax_amount", "absent", warning="vat-label-not-present"))
        fields.append(_field("tax_rates", "absent", warning="vat-rate-label-not-present"))
    elif len(iva_labels) > 1:
        fields.append(_field("tax_amount", "ambiguous", warning="duplicate-vat-label"))
        fields.append(_field("tax_rates", "ambiguous", warning="duplicate-vat-label"))
    else:
        label = iva_labels[0]
        rate_spans = [hit for hit in _matches(tokens, r"\d{1,3}(?:[.,]\d{1,2})")
                      if 0.58 <= (hit.bounds.x0 + hit.bounds.x1) / 2 / page.width <= .78
                      and .79 <= (hit.bounds.top + hit.bounds.bottom) / 2 / page.height <= .91]
        pct_spans = [hit for hit in _matches(tokens, r"%")
                     if .58 <= (hit.bounds.x0 + hit.bounds.x1) / 2 / page.width <= .78
                     and .79 <= (hit.bounds.top + hit.bounds.bottom) / 2 / page.height <= .91]
        paired_rates = []
        for rate in rate_spans:
            nearby = [pct for pct in pct_spans if abs(pct.bounds.top - rate.bounds.top) < 3
                      and abs(pct.bounds.x0 - rate.bounds.x1) < 20]
            if nearby:
                paired_rates.append((rate, nearby[0]))
        if len(paired_rates) == 1:
            rate, pct = paired_rates[0]
            combined = Evidence(1, rate.text + pct.text,
                                BoundingBox(rate.bounds.x0, min(rate.bounds.top, pct.bounds.top),
                                            pct.bounds.x1, max(rate.bounds.bottom, pct.bounds.bottom)),
                                rate.token.id, " ".join(span.text for span in label[1]), label[2],
                                "summary-iva-rate", extractor.id, extractor.version,
                                rate.token_ids + pct.token_ids)
            fields.append(_field("tax_rates", "confirmed", value=(_number(rate.text),), evidence=(combined,)))
        elif len(paired_rates) > 1:
            fields.append(_field("tax_rates", "ambiguous", candidates=tuple(_number(r.text) for r, _ in paired_rates),
                                 warning="multiple-vat-rate-values-in-summary-cell"))
        else:
            fields.append(_field("tax_rates", "unreadable", warning="iva-rate-percent-not-located"))
        iva_cell = _cell_limits(label, all_labels, page)
        amounts = [hit for hit in _matches(tokens, r"-?\d{1,3}(?:\.\d{3})*(?:,\d{2})|-?\d+(?:,\d{2})?")
                   if _number(hit.text) is not None and "%" not in hit.text
                   and iva_cell[0] <= (hit.bounds.x0 + hit.bounds.x1) / 2 / page.width <= iva_cell[1]
                   and .79 <= (hit.bounds.top + hit.bounds.bottom) / 2 / page.height <= .91]
        amounts = [hit for hit in amounts if (hit.bounds.top + hit.bounds.bottom) / 2 > label[2].bottom
                   and not any(abs((hit.bounds.x0 + hit.bounds.x1) - (rate.bounds.x0 + rate.bounds.x1)) < 24
                               and abs(hit.bounds.top - rate.bounds.top) < 3 for rate, _ in paired_rates)]
        if len(amounts) == 1:
            fields.append(_field("tax_amount", "confirmed", value=_number(amounts[0].text),
                                 evidence=(_evidence(amounts[0], label, "summary-iva-amount-column", extractor),)))
        elif len(amounts) > 1:
            fields.append(_field("tax_amount", "conflict", candidates=tuple(_number(h.text) for h in amounts),
                                 warning="multiple-iva-amounts-in-summary-cell"))
        else:
            fields.append(_field("tax_amount", "insufficient_evidence", warning="iva-amount-not-located"))

    total_labels = [hit for hit in all_labels if hit[0] == "TOTAL EUROS"]
    if total_labels:
        candidate = amount_field("displayed_total_euros", "TOTAL EUROS", x_band=(.75, .98))
        fields.append(ExtractedField(candidate.name, candidate.value, candidate.evidence,
                                     candidate.warnings, "unverified_semantics", candidate.candidates))
    else:
        fields.append(_field("displayed_total_euros", "absent", warning="total-euros-label-not-present"))
    fields.extend((
        _field("total_invoice", "ambiguous", warning="TOTAL-EUROS-has-no-confirmed-financial-role"),
        _field("total_due", "ambiguous", warning="TOTAL-EUROS-has-no-confirmed-financial-role"),
    ))
    return tuple(fields)


def _extract_template(pages, token_pages, template, extractor):
    if len(pages) != len(token_pages) or not pages:
        return (), ("token-page-count-mismatch",)
    header_zone = (0.04, 0.27, 0.49 if template == "T01" else 0.72, 0.34)
    tokens = token_pages[0]
    fields = [
        _header_field("invoice_number", "NO DE FACTURA", r"[A-Z]*\d{5,10}", tokens, pages[0], extractor,
                      header_zone, str),
        _header_field("issue_date", "FECHA", r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", tokens, pages[0], extractor,
                      header_zone, _parse_date),
    ]
    fields.append(_header_field("client_code", "CODIGO CLIENTE", r"\d{4,8}", tokens, pages[0], extractor,
                                 header_zone, str))
    if template == "T01":
        # The explicit TOTAL EUROS label is evidence of currency, not of either financial total.
        euros = _labels(tokens, pages[0], {"TOTAL EUROS"}, (0.04, .77, .98, .92))
        if len(euros) == 1:
            label = euros[0]
            word = next((span for span in label[1] if _norm(span.text) == "EUROS"), label[1][-1])
            fields.append(_field("currency", "confirmed", value="EUR",
                                 evidence=(_evidence(word, label, "explicit-euros-label", extractor),)))
        elif not euros:
            fields.append(_field("currency", "absent", warning="EUROS-label-not-present"))
        else:
            fields.append(_field("currency", "ambiguous", warning="duplicate-EUROS-label"))
    else:
        euro = [hit for hit in _matches(tokens, "€")
                if .48 <= (hit.bounds.x0 + hit.bounds.x1) / 2 / pages[0].width <= .68
                and .34 <= (hit.bounds.top + hit.bounds.bottom) / 2 / pages[0].height <= .69]
        if euro:
            fields.append(_field("currency", "confirmed", value="EUR",
                                 evidence=(_evidence(euro[0], None, "explicit-euro-symbol", extractor),)))
        else:
            fields.append(_field("currency", "absent", warning="explicit-currency-symbol-not-present"))
    fields.extend(_tax_ids(tokens, pages[0], extractor))
    fields.extend(_summary_fields(pages, token_pages, extractor, template))
    warnings = tuple(f"{field.name}:{field.status}:{field.warnings[0]}" for field in fields if field.warnings)
    return tuple(fields), warnings


def _parse_date(text):
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def _tax_ids(tokens, page, extractor):
    receiver_zone = (0.05, 0.10, 0.52, 0.29)
    receiver_labels = _labels(tokens, page, {"DIRECCION FISCAL"}, (.04, .10, .52, .36))
    receiver_label = receiver_labels[0] if len(receiver_labels) == 1 else None
    receiver_ids = [hit for hit in _matches(tokens, r"[A-Z]\d{8}")
                    if not hit.token.id.endswith("-vertical")
                    and receiver_zone[0] <= (hit.bounds.x0 + hit.bounds.x1) / 2 / page.width <= receiver_zone[2]
                    and receiver_zone[1] <= (hit.bounds.top + hit.bounds.bottom) / 2 / page.height <= receiver_zone[3]]
    if len(receiver_ids) == 1:
        receiver = _field("receiver_tax_id", "confirmed", value=receiver_ids[0].text,
                          evidence=(_evidence(receiver_ids[0], receiver_label,
                                              "fiscal-recipient-region", extractor),))
    elif not receiver_ids:
        receiver = _field("receiver_tax_id", "absent", warning="no-id-in-fiscal-recipient-region")
    else:
        receiver = _field("receiver_tax_id", "ambiguous", warning="multiple-ids-in-fiscal-recipient-region")

    issuer = []
    for token in tokens:
        if "-v" not in token.id:
            continue
        cif = re.search(r"C\.?I\.?F\.?", token.text, re.IGNORECASE)
        if not cif:
            continue
        id_match = re.search(r"[A-Z]-?\d{8}", token.text[cif.end():], re.IGNORECASE)
        if id_match:
            start, end = cif.end() + id_match.start(), cif.end() + id_match.end()
            id_span = _span(token, start, end)
            label_span = _span(token, cif.start(), cif.end())
            issuer.append((id_span, label_span))
    if len(issuer) == 1:
        issuer_value, issuer_label = issuer[0]
        issuer_field = _field("issuer_tax_id", "confirmed", value=_norm(issuer_value.text).replace(" ", ""),
                              evidence=(_evidence(issuer_value, ("CIF", (issuer_label,), issuer_label.bounds),
                                                  "CIF-label-in-vertical-issuer-text", extractor),))
    elif not issuer:
        issuer_field = _field("issuer_tax_id", "unreadable", warning="issuer-CIF-not-localized-in-legal-margin")
    else:
        issuer_field = _field("issuer_tax_id", "ambiguous", warning="multiple-issuer-CIF-spans")
    return (issuer_field, receiver)


def extract_t01_header_summary(pages, token_pages):
    return _extract_template(pages, token_pages, "T01", template_extractor_for("T01"))


def extract_t02_header_summary(pages, token_pages):
    return _extract_template(pages, token_pages, "T02", template_extractor_for("T02"))


_LINE_HEADERS = {
    "T01": ("CONCEPTO", "ALBARAN", "FECHA", "T.CHEQUES", "TOTAL"),
    "T02": ("DESCRIPCION", "CANTIDAD", "IMP. UNIDAD", "IMPORTE", "TIPO IVA", "CUOTA IVA", "TOTAL"),
}


def _line_header(tokens, page, template):
    expected = _LINE_HEADERS[template]
    body = [token for token in tokens if "-v" not in token.id]
    rows = _rows(tokens, page, tolerance=8)
    anchors = [token for token in body if _norm(token.text) == expected[0]]
    for anchor in anchors:
        center_y = (anchor.bounds.top + anchor.bounds.bottom) / 2
        neighborhood = sorted((token for token in body
                               if abs((token.bounds.top + token.bounds.bottom) / 2 - center_y) <= 13),
                              key=lambda token: (token.bounds.x0, token.bounds.top))
        spans = []
        for label in expected:
            if template == "T02" and label == "TIPO IVA":
                group = [token for token in neighborhood
                         if 0.67 <= token.bounds.x0 / page.width <= 0.76
                         and _norm(token.text) in {"T", "I", "P", "O", "V", "A"}]
                if group:
                    spans.append((label, min(t.bounds.x0 for t in group), max(t.bounds.x1 for t in group)))
                    continue
            target = _norm(label).replace(" ", "")
            matches = []
            for start in range(len(neighborhood)):
                for end in range(start + 1, min(len(neighborhood), start + 6) + 1):
                    group = neighborhood[start:end]
                    if max(item.bounds.top for item in group) - min(item.bounds.top for item in group) > 13:
                        continue
                    if _norm(" ".join(item.text for item in group)).replace(" ", "") == target:
                        matches.append(group)
            if len(matches) != 1:
                break
            group = matches[0]
            spans.append((label, min(t.bounds.x0 for t in group), max(t.bounds.x1 for t in group)))
        if len(spans) == len(expected):
            end_row = max(index for index, row in enumerate(rows)
                          if any(abs((token.bounds.top + token.bounds.bottom) / 2 - center_y) <= 13
                                 for token in row))
            return end_row, tuple(spans), None
    return None, (), "detail-header-not-found"


def _line_decimal(text):
    raw = text.strip().replace("€", "").replace("EUR", "").replace(" ", "")
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?", raw):
        return None
    try:
        return Decimal(raw.replace(".", "").replace(",", "."))
    except InvalidOperation:
        return None


def extract_template_lines(pages, token_pages, template):
    """Offline geometric row reconstruction; unsupported layouts abstain explicitly."""
    if template not in _LINE_HEADERS:
        return (), ("unsupported-template",)
    if len(pages) != 1 or len(token_pages) != 1:
        return (), ("multipage-detail-not-supported",)
    page, tokens = pages[0], token_pages[0]
    header_index, labels, warning = _line_header(tokens, page, template)
    if warning:
        return (), (warning,)
    labels = sorted(labels, key=lambda item: item[1])
    boundaries = [0.0] + [right[1] if template == "T02" and index == 0 else (left[2] + right[1]) / 2
                          for index, (left, right) in enumerate(zip(labels, labels[1:]))] + [page.width]
    detail_end = page.height * (0.75 if template == "T01" else 0.71)
    rows = _rows(tokens, page, tolerance=8)
    header_y = min((label[1] for label in labels), default=0)
    from application.extractor_registry import template_line_extractor_for
    extractor = template_line_extractor_for(template)
    lines = []
    pending_description = []
    pending_evidence = []
    previous_y = header_y
    for row in rows[header_index + 1:]:
        y = min(t.bounds.top for t in row)
        if y > detail_end:
            break
        text = " ".join(t.text for t in row).strip()
        normalized = _norm(text)
        if re.search(r"\b(SUBTOTAL|BASE IMPONIBLE|TOTAL FACTURA|TOTAL A PAGAR)\b", normalized):
            break
        if template == "T01" and re.search(r"\b(COMISION|FORFAIT|GASTOS ENVIO)\b", normalized):
            break
        note_tokens = [token for token in row
                       if (token.bounds.x0 + token.bounds.x1) / 2 < boundaries[1]]
        note_text = " ".join(token.text for token in note_tokens).strip()
        if template == "T02" and "ANULA FACTURA" in _norm(note_text):
            note_evidence = tuple(_evidence(_span(token, 0, len(token.text)), None,
                "document-cancellation-note", extractor) for token in note_tokens)
            lines.append(ProposedLine(note_text, {"row_type": "cancellation", "document_note": True},
                                      note_evidence, ("document-cancellation-note",)))
            previous_y = y
            continue
        cells = [[] for _ in labels]
        for token in row:
            center = (token.bounds.x0 + token.bounds.x1) / 2
            column = next((i for i in range(len(labels)) if boundaries[i] <= center < boundaries[i + 1]), None)
            if column is not None:
                cells[column].append(token)
        desc_tokens = cells[0]
        numeric = {}
        for index, label in enumerate(labels[1:], start=1):
            if template == "T01" and label[0] == "FECHA":
                date_tokens = [token for token in cells[index] if _parse_date(token.text) is not None]
                if len(date_tokens) == 1:
                    numeric[label[0]] = (_parse_date(date_tokens[0].text), date_tokens[0])
                continue
            values = [(_line_decimal(token.text), token) for token in cells[index]]
            values = [(value, token) for value, token in values if value is not None]
            if len(values) > 1:
                return tuple(lines), ("multiple-values-in-column",)
            if values:
                numeric[label[0]] = values[0]
        description = " ".join(token.text for token in desc_tokens).strip()
        if template == "T02" and _norm(description).startswith("NUMERO DE BENEFICIARIOS"):
            ev = tuple(_evidence(_span(token, 0, len(token.text)), None, "detail-description-column", extractor)
                       for token in desc_tokens)
            values = {"row_type": "informational", "row_order": len(lines) + 1}
            if "CANTIDAD" in numeric:
                values["quantity"] = numeric["CANTIDAD"][0]
                _, token = numeric["CANTIDAD"]
                ev += (_evidence(_span(token, 0, len(token.text)), None, "detail-column-center", extractor),)
            lines.append(ProposedLine(description, values, ev,
                                      ("informational-row-no-charge",)))
            previous_y = y
            continue
        if not numeric:
            if description:
                if pending_description and y - previous_y > 18:
                    lines.append(ProposedLine(" ".join(pending_description), {"row_type": "unknown",
                        "row_order": len(lines) + 1}, tuple(pending_evidence), ("unassociated-description-band",)))
                    pending_description, pending_evidence = [], []
                pending_description.append(description)
                pending_evidence.extend(_evidence(_span(token, 0, len(token.text)), None,
                    "detail-description-column", extractor) for token in desc_tokens)
            previous_y = y
            continue
        if not description and lines and y - previous_y <= 24:
            prior = lines[-1]
            values = {**prior.values}
            values.update({key: number for key, (number, _) in numeric.items()})
            evidence = prior.evidence + tuple(_evidence(_span(token, 0, len(token.text)), None,
                "detail-column-center", extractor) for _, token in numeric.values())
            lines[-1] = ProposedLine(prior.description, values, evidence, prior.warnings)
            previous_y = y
            continue
        if description:
            if pending_description:
                lines.append(ProposedLine(" ".join(pending_description),
                    {"row_type": "informational", "document_note": True}, tuple(pending_evidence),
                    ("unpriced-description-band", "ambiguous-description-prefix")))
                pending_description, pending_evidence = [], []
            pending_description.append(description)
            pending_evidence.extend(_evidence(_span(token, 0, len(token.text)), None,
                "detail-description-column", extractor) for token in desc_tokens)
        desc = " ".join(pending_description) or None
        evidence = list(pending_evidence)
        values = {"row_order": len(lines) + 1, "row_type": "unknown"}
        for label, (number, token) in numeric.items():
            if template == "T01":
                key = {"ALBARAN": "reference", "FECHA": "line_date", "T.CHEQUES": "t_cheques_unverified",
                       "TOTAL": "documentary_total"}.get(label)
            else:
                key = {"CANTIDAD": "quantity", "IMP. UNIDAD": "unit_price", "IMPORTE": "amount",
                       "TIPO IVA": "vat_rate", "CUOTA IVA": "vat_amount", "TOTAL": "total"}.get(label)
            if key:
                values[key] = number
                evidence.append(_evidence(_span(token, 0, len(token.text)), None,
                    "detail-column-center", extractor))
        if template == "T02" and "amount" in values:
            values["row_type"] = "discount" if _norm(desc or "").startswith("DESCUENTO") else "charge"
        elif template == "T01" and "documentary_total" in values:
            values["row_type"] = "charge"
        if re.search(r"\bANULA\b|\bCANCELA", _norm(desc or "")):
            values["row_type"] = "cancellation"
        lines.append(ProposedLine(desc, values, tuple(evidence)))
        pending_description, pending_evidence = [], []
        previous_y = y
    if pending_description and template == "T02":
        lines.append(ProposedLine(" ".join(pending_description), {"row_type": "unknown",
            "row_order": len(lines) + 1}, tuple(pending_evidence), ("unassociated-description-band",)))
    if template == "T01":
        lines = [line for line in lines if line.description and line.values.get("row_type") == "charge"]
    return tuple(lines), ()


def extract_classified_header_summary(pages, token_pages, classification=None) -> TemplateExtraction:
    classification = classification or detect_template(pages)
    if classification.template == "unknown":
        return TemplateExtraction(classification, None, None, (), (classification.reason or "unknown-template",))
    extractor = template_extractor_for(classification.template)
    fields, _, warnings = extractor.extract(pages, token_pages)
    return TemplateExtraction(classification, extractor.id, extractor.version, fields, warnings)


def extract_as400_header_summary(data: bytes) -> TemplateExtraction:
    """Run only a positively classified AS/400 extractor; unknown never falls back."""
    pages = extract_pdf(data)
    return extract_classified_header_summary(pages, extract_pdf_tokens(data))
