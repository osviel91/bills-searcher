import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

from domain.structured_extraction import Evidence, ExtractedField, PDFPage, ProposedLine


def _money(value):
    value = value.strip().replace("€", "").replace("EUR", "").replace(" ", "")
    if "," in value:
        value = value.replace(".", "").replace(",", ".")
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _dates(value):
    for pattern, fmt in (("%d/%m/%Y", r"\d{1,2}/\d{1,2}/\d{4}"),
                         ("%d-%m-%Y", r"\d{1,2}-\d{1,2}-\d{4}"),
                         ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
                         ("%d.%m.%Y", r"\d{1,2}\.\d{1,2}\.\d{4}")):
        match = re.search(fmt, value)
        if match:
            try:
                return datetime.strptime(match.group(), pattern).date(), match.group()
            except ValueError:
                pass
    return None, None


def extract_invoice(pages: tuple[PDFPage, ...]):
    lines = [(page, block) for page in pages for block in page.blocks]
    fields = {name: ExtractedField(name) for name in (
        "invoice_number", "issue_date", "issuer_id", "customer_id", "issuer_tax_id",
        "customer_tax_id", "currency", "tax_base", "vat", "total",
    )}
    warnings = []

    def put(name, value, page, quote, block):
        if value is not None:
            fields[name] = ExtractedField(name, value, (Evidence(page.number, quote, block.bounds, block.id),))

    patterns = {
        "invoice_number": r"(?:n[uú]m(?:ero)?\.?\s*(?:de\s*)?factura|factura(?:\s*n[uú]m(?:ero)?\.?)?)\s*[:#-]?\s*([\w/-]+)",
        "issuer_id": r"(?:emisor|proveedor)\s*(?:id|c[oó]digo)?\s*[:#-]\s*([^\s,;]+)",
        "customer_id": r"(?:cliente|destinatario)\s*(?:id|c[oó]digo)?\s*[:#-]\s*([^\s,;]+)",
    }
    for page, block in lines:
        text = block.text
        for name, pattern in patterns.items():
            if fields[name].value is None:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and (name != "invoice_number" or re.search(r"\d", match.group(1))):
                    put(name, match.group(1), page, match.group(), block)
        if fields["issue_date"].value is None and re.search(r"fecha", text, re.IGNORECASE):
            value, quote = _dates(text)
            put("issue_date", value, page, quote, block)
        if fields["currency"].value is None:
            match = re.search(r"\b(EUR|USD|GBP)\b|€|\$|£", text, re.IGNORECASE)
            if match:
                put("currency", {"€": "EUR", "$": "USD", "£": "GBP"}.get(match.group(), match.group().upper()),
                    page, match.group(), block)
        for name, label in (("tax_base", r"base\s+imponible"),
                            ("vat", r"(?:cuota\s+(?:de\s+)?iva|iva(?:\s+\d+(?:[.,]\d+)?\s*%)?)"),
                            ("total", r"total(?:\s+factura)?")):
            if fields[name].value is not None:
                continue
            match = re.search(rf"({label})\s*[:\-]?\s*(-?[\d.]+(?:,\d{{2}})?|-?[\d,]+(?:\.\d{{2}})?)", text,
                              re.IGNORECASE)
            if match:
                put(name, _money(match.group(2)), page, match.group(), block)
        ids = re.findall(r"\b[A-Z]\d{7,8}[A-Z0-9]?\b", text, re.IGNORECASE)
        if ids:
            if fields["issuer_tax_id"].value is None and re.search(r"(?:emisor|proveedor|cif|nif)", text, re.I):
                put("issuer_tax_id", ids[0], page, text, block)
            elif fields["customer_tax_id"].value is None and re.search(r"(?:cliente|destinatario)", text, re.I):
                put("customer_tax_id", ids[0], page, text, block)

    found = next(((index, page, block) for index, (page, block) in enumerate(lines)
                  if re.search(r"descripci[oó]n|concepto", block.text, re.I)), None)
    result_lines = []
    if found:
        header_index, page, header = found
        header_cols = re.split(r"\s{2,}|\|", header.text.strip("| "))
        if len(header_cols) < 2:
            header_cols = []
        description_parts, description_evidence = [], []
        for row_page, row in lines[header_index + 1:]:
            if row_page.number != page.number or re.search(r"total|base imponible|iva", row.text, re.I):
                break
            cells = re.split(r"\s{2,}|\|", row.text.strip("| "))
            if not cells:
                continue
            values = {}
            for key, cell in zip(header_cols[1:], cells[1:]):
                key = key.strip().lower()
                if any(word in key for word in ("cant", "cantidad")) and re.fullmatch(r"\d+(?:[.,]\d+)?", cell.strip()):
                    values["quantity"] = Decimal(cell.strip().replace(",", "."))
                elif any(word in key for word in ("precio", "importe", "total")):
                    amount = _money(cell)
                    if amount is not None:
                        values["amount" if "importe" in key or "total" in key else "unit_price"] = amount
            description_parts.append(cells[0].strip())
            description_evidence.append(Evidence(row_page.number, row.text, row.bounds, row.id))
            if values:
                result_lines.append(ProposedLine(" ".join(description_parts) or None, values,
                                                 tuple(description_evidence)))
                description_parts = []
                description_evidence = []
        if description_parts:
            result_lines.append(ProposedLine(" ".join(description_parts), {}, tuple(description_evidence)))
    return tuple(fields.values()), tuple(result_lines), tuple(warnings)


GENERIC_EXTRACTOR_ID = "generic-invoice"
GENERIC_EXTRACTOR_VERSION = "1"
