from dataclasses import dataclass

from domain.invoice_extraction import GENERIC_EXTRACTOR_ID, GENERIC_EXTRACTOR_VERSION, extract_invoice


@dataclass(frozen=True)
class Extractor:
    id: str
    version: str
    priority: int
    detector: object
    extract: object


GENERIC = Extractor(GENERIC_EXTRACTOR_ID, GENERIC_EXTRACTOR_VERSION, 0, lambda pages: True, extract_invoice)
EXTRACTORS = (GENERIC,)


@dataclass(frozen=True)
class TemplateDetector:
    id: str
    version: str
    headers: tuple[str, ...]
    labels: tuple[str, ...]
    geometry: str


TEMPLATE_DETECTORS = (
    TemplateDetector(
        "as400-t01", "1", ("CONCEPTO", "ALBARAN", "FECHA", "T.CHEQUES", "TOTAL"),
        ("COMISION", "FORFAIT GESTION", "GASTOS ENVIO", "TOTAL EUROS"), "split-detail-band",
    ),
    TemplateDetector(
        "as400-t02", "1",
        ("DESCRIPCION", "CANTIDAD", "IMP. UNIDAD", "IMPORTE", "TIPO IVA", "CUOTA IVA", "TOTAL"),
        ("BASE IMPONIBLE", "CUOTA IVA", "TOTAL FACTURA", "TOTAL A PAGAR"), "single-detail-band",
    ),
)


def _template_matches(template):
    def detector(pages):
        from application.template_detection import detect_template
        return detect_template(pages).template == template
    return detector


def _extract_t01(pages, tokens):
    from application.template_extraction import extract_t01_header_summary
    fields, warnings = extract_t01_header_summary(pages, tokens)
    return fields, (), warnings


def _extract_t02(pages, tokens):
    from application.template_extraction import extract_t02_header_summary
    fields, warnings = extract_t02_header_summary(pages, tokens)
    return fields, (), warnings


# Offline-only records; deliberately excluded from the active EXTRACTORS tuple.
TEMPLATE_EXTRACTORS = (
    Extractor("as400-t01-header-summary", "1", 1, _template_matches("T01"), _extract_t01),
    Extractor("as400-t02-header-summary", "1", 1, _template_matches("T02"), _extract_t02),
)


def template_extractor_for(template):
    for extractor in TEMPLATE_EXTRACTORS:
        if extractor.id == f"as400-{template.lower()}-header-summary":
            return extractor
    raise ValueError(f"No offline template extractor for {template}")


def select_extractor(pages, extractors=EXTRACTORS):
    matched = [item for item in extractors if item.detector(pages)]
    if not matched:
        return GENERIC, ("No extractor matched; generic fallback selected",)
    priority = max(item.priority for item in matched)
    winners = [item for item in matched if item.priority == priority]
    if len(winners) != 1:
        return GENERIC, ("Ambiguous extractors at priority " + str(priority) + ": " +
                         ", ".join(sorted(item.id for item in winners)) + "; generic fallback selected",)
    return winners[0], ()
