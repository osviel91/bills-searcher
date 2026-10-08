import re
from datetime import datetime


def extract_invoice_metadata(text: str) -> dict:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    content = "\n".join(line for line in lines if line)
    result = {
        "invoice_number": None, "invoice_date": None, "customer_code": None,
        "customer_tax_id": None, "delivery_note_ids": [], "concepts": [],
        "currency": "EUR" if "€" in text or "EUROS" in text.upper() else None,
        "total_payable": None, "invoice_total": None, "tax_base": None, "vat": None,
        "confidence": {}, "evidence": {},
    }

    def set_field(name, value, evidence, confidence=0.95):
        if value is not None:
            result[name] = value
            result["confidence"][name] = confidence
            result["evidence"][name] = evidence[:300]

    def cells(line):
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def invoice_id(value):
        return (re.fullmatch(r"(?=[A-Z0-9/-]*\d)[A-Z0-9/-]{5,}", value, re.IGNORECASE) and
                not re.fullmatch(r"[ABCDEFGHJNPQRSUVW]\d{7}[A-J0-9]", value, re.IGNORECASE))

    for index, line in enumerate(lines):
        if "Nº DE FACTURA" not in line.upper() and "N° DE FACTURA" not in line.upper():
            continue
        labels = [cell.upper() for cell in cells(line)]
        rows = [cells(candidate) for candidate in reversed(lines[:index])
                if "|" in candidate and not re.fullmatch(r"[|\s:-]+", candidate)][:1]
        rows.extend(cells(candidate) for candidate in lines[index + 1:index + 6]
                    if "|" in candidate and not re.fullmatch(r"[|\s:-]+", candidate))
        for row in rows:
            for column, label in enumerate(labels):
                if column >= len(row):
                    continue
                value = row[column]
                if "FECHA" in label and re.fullmatch(r"\d{2}[./-]\d{2}[./-]\d{4}", value):
                    try:
                        normalized = datetime.strptime(value.replace("/", ".").replace("-", "."), "%d.%m.%Y").date().isoformat()
                    except ValueError:
                        continue
                    set_field("invoice_date", normalized, value)
                if "CODIGO CLIENTE" in label and re.fullmatch(r"(?:[A-Z]?\d{4,})", value, re.IGNORECASE):
                    set_field("customer_code", value, value)
            if row and invoice_id(row[0]):
                set_field("invoice_number", row[0], row[0])
        prior = [candidate for candidate in lines[:index] if candidate and "|" not in candidate and
                 not re.fullmatch(r"[-\s]+", candidate)]
        if prior and invoice_id(prior[-1]):
            set_field("invoice_number", prior[-1], prior[-1])
        break

    number = re.search(
        r"(?:N[º°.]\s*DE\s*FACTURA|N[º°.]\s*FACTURA)\s*:?\s*([A-Z0-9][A-Z0-9/-]*)",
        content, re.IGNORECASE,
    )
    if number and number.group(1).upper() not in {"FECHA", "CODIGO", "CÓDIGO", "N", "PEDIDO", "PROVEEDOR"}:
        set_field("invoice_number", number.group(1), number.group(0))
    else:
        number = re.search(
            r"\b([A-Z]?[A-Z0-9/-]{4,})\s+\d{2}[./-]\d{2}[./-]\d{4}\s+\d{4,}\s+N[º°.]\s*DE\s*FACTURA",
            content, re.IGNORECASE,
        )
        if number:
            set_field("invoice_number", number.group(1), number.group(0))

    date = re.search(r"(?:FECHA\s*:?[ \t]*|\b)(\d{2}[./-]\d{2}[./-]\d{4})\b", content, re.IGNORECASE)
    if date:
        raw_date = date.group(1)
        try:
            iso_date = datetime.strptime(raw_date.replace("/", ".").replace("-", "."), "%d.%m.%Y").date().isoformat()
        except ValueError:
            iso_date = None
        set_field("invoice_date", iso_date, date.group(0))

    client_row = re.search(
        r"([A-Z]?[A-Z0-9/-]{4,})\s+\d{2}[./-]\d{2}[./-]\d{4}\s+(\d{4,})[^\n]*N[º°.]\s*DE\s*FACTURA",
        content, re.IGNORECASE,
    )
    if client_row:
        set_field("customer_code", client_row.group(2), client_row.group(0))
    else:
        code = re.search(r"CODIGO\s+CLIENTE\s*:?[ \t]*([A-Z0-9-]+)", content, re.IGNORECASE)
        if code and code.group(1).upper() not in {"N", "Nº", "PEDIDO", "FECHA", "PROVEEDOR"}:
            set_field("customer_code", code.group(1), code.group(0))

    # ponytail: accept the adjacent ID/date/client-code layout when PDF table extraction loses columns.
    for index, line in enumerate(lines):
        if not re.fullmatch(r"\d{2}[./-]\d{2}[./-]\d{4}", line):
            continue
        before = next((candidate for candidate in reversed(lines[:index]) if candidate), "")
        after = next((candidate for candidate in lines[index + 1:] if candidate), "")
        if invoice_id(before) and re.fullmatch(r"[A-Z]?\d{5,}", before, re.IGNORECASE):
            set_field("invoice_number", before, before, 0.6)
        if re.fullmatch(r"\d{4,}", after):
            set_field("customer_code", after, after, 0.6)

    issuer = re.search(r"C\.?I\.?F\.?\s*:\s*([A-Z]\d{7,8}[A-Z0-9]?)", content, re.IGNORECASE)
    tax_ids = re.findall(r"\b[ABCDEFGHJNPQRSUVW]\d{7}[A-J0-9]\b", content, re.IGNORECASE)
    customer_ids = [tax_id for tax_id in tax_ids if not issuer or tax_id.upper() != issuer.group(1).upper()]
    if customer_ids:
        evidence = next((line for line in lines if customer_ids[0] in line), customer_ids[0])
        set_field("customer_tax_id", customer_ids[0], evidence, 0.85 if issuer else 0.65)

    header = re.search(r"\b(?:DESCRIPCION|CONCEPTO)\b", content, re.IGNORECASE)
    if header:
        table = content[header.end():]
        table = re.split(r"\b(?:SUBTOTAL|FORMA DE PAGO|BASE IMPONIBLE)\b", table, maxsplit=1, flags=re.IGNORECASE)[0]
        concepts = []
        for line in table.splitlines():
            line = re.sub(r"\s+", " ", line).strip(" :-")
            if line and not re.fullmatch(r"[\d., €%/-]+", line, re.IGNORECASE):
                concepts.append(line)
        if concepts:
            result["concepts"] = concepts
            result["confidence"]["concepts"] = 0.8
            result["evidence"]["concepts"] = "\n".join(concepts)[:500]

    delivery_header = re.search(r"\bALBARAN\b", content, re.IGNORECASE)
    if delivery_header:
        delivery_ids = re.findall(r"\b\d{4,}\b", content[delivery_header.end():delivery_header.end() + 180])
        if delivery_ids:
            result["delivery_note_ids"] = [delivery_ids[0]]
            result["confidence"]["delivery_note_ids"] = 0.8
            result["evidence"]["delivery_note_ids"] = delivery_ids[0]

    amount_pattern = re.compile(r"(?<!\w)-?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{2}|\.\d{2})")
    amount_labels = (("tax_base", re.compile(r"BASE\s+IMPONIBLE", re.IGNORECASE)),
                     ("vat", re.compile(r"CUOTA\s+IVA", re.IGNORECASE)),
                     ("invoice_total", re.compile(r"TOTAL\s+FACTURA", re.IGNORECASE)),
                     ("total_payable", re.compile(r"TOTAL\s+A\s+PAGAR", re.IGNORECASE)))
    for index, line in enumerate(lines):
        if "|" in line:
            labels = [cell.upper() for cell in cells(line)]
            if any(label in {"BASE IMPONIBLE", "TOTAL FACTURA", "TOTAL A PAGAR"} or "CUOTA IVA" in label
                   for label in labels):
                value_row = next((cells(candidate) for candidate in lines[index + 1:]
                                  if "|" in candidate and not re.fullmatch(r"[|\s:-]+", candidate)), [])
                for field, label_pattern in amount_labels:
                    column = next((column for column, label in enumerate(labels)
                                   if label_pattern.search(label)), None)
                    if column is None or column >= len(value_row):
                        continue
                    match = amount_pattern.search(value_row[column])
                    if match:
                        raw = match.group().replace(".", "").replace(",", ".") if "," in match.group() else match.group().replace(",", "")
                        set_field(field, float(raw), f"{line} {value_row[column]}")
                if "CUOTA IVA" in content.upper() and "vat" not in result["confidence"]:
                    base_column = next((i for i, label in enumerate(labels) if "BASE IMPONIBLE" in label), None)
                    total_column = next((i for i, label in enumerate(labels) if "TOTAL FACTURA" in label), None)
                    if base_column is not None and total_column is not None:
                        vat_value = next((amount_pattern.search(value_row[i]) for i in range(base_column + 1, min(total_column, len(value_row)))
                                          if amount_pattern.search(value_row[i])), None)
                        if vat_value:
                            raw = vat_value.group().replace(".", "").replace(",", ".") if "," in vat_value.group() else vat_value.group().replace(",", "")
                            set_field("vat", float(raw), f"Cuota IVA {vat_value.group()}", 0.8)
        headings = [(field, match.start()) for field, label in amount_labels
                    if (match := label.search(line))]
        if not headings or index + 1 == len(lines):
            continue
        values = list(amount_pattern.finditer(lines[index + 1]))
        if len(values) < len(headings):
            continue
        headings.sort(key=lambda item: item[1])
        for position, (field, _) in enumerate(headings):
            match = values[position]
            raw = match.group().replace(".", "").replace(",", ".") if "," in match.group() else match.group().replace(",", "")
            set_field(field, float(raw), f"{line} {lines[index + 1]}")

    for field, label in (("total_payable", r"TOTAL\s+A\s+PAGAR"), ("invoice_total", r"TOTAL\s+FACTURA"),
                         ("tax_base", r"BASE\s+IMPONIBLE"), ("vat", r"CUOTA\s+IVA")):
        if field in result["confidence"]:
            continue
        match = re.search(rf"{label}[^\n\d-]*(-?[\d.]+,\d{{2}}|-?[\d,]+\.\d{{2}})", content, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(".", "").replace(",", ".") if "," in match.group(1) else match.group(1).replace(",", "")
            set_field(field, float(raw), match.group(0), 0.65)

    return result
