import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import patch

from application.extractor_registry import EXTRACTORS, TEMPLATE_EXTRACTORS
from application.template_extraction import (extract_as400_header_summary, extract_t01_header_summary,
                                             extract_t02_header_summary, extract_template_lines)
from domain.structured_extraction import BoundingBox, PDFPage, PDFToken, TableRegion


def token(text, x, y, width=None, *, token_id=None, vertical=False):
    width = width if width is not None else max(5, len(text) * 5)
    chars = []
    for index, char in enumerate(text):
        if vertical:
            chars.append(BoundingBox(x, y + index, x + 4, y + index + 1))
        else:
            left = x + width * index / max(len(text), 1)
            right = x + width * (index + 1) / max(len(text), 1)
            chars.append(BoundingBox(left, y, right, y + 10))
    bounds = BoundingBox(min(c.x0 for c in chars), min(c.top for c in chars),
                         max(c.x1 for c in chars), max(c.bottom for c in chars))
    return PDFToken(token_id or f"token-{x}-{y}-{text}", text, bounds,
                    tuple(f"char-{i}" for i in range(len(text))), tuple(chars))


def t02_fixture(*, dates=("03.09.2026",), summary=True, duplicate_date=False, rates=("21,00",),
                base="100,00", tax_amount="21,00", total="121,00"):
    page = PDFPage(1, 1000, 1000, "")
    tokens = [
        token("12345678", 70, 295, 70), token("NO", 55, 325), token("DE", 65, 325), token("FACTURA", 75, 325),
        token(dates[0], 230, 295, 70), token("FECHA", 235, 325),
        token("0001234", 410, 295, 45), token("CODIGO", 390, 325), token("CLIENTE", 425, 325),
        token("B12345678", 80, 170, 45),
        token("C.I.F.:A87654321", 25, 100, 10, token_id="p1-v1", vertical=True),
        token("€", 580, 410),
    ]
    if duplicate_date:
        tokens.extend((token("FECHA", 510, 325), token("04.09.2026", 510, 295, 70)))
    if len(dates) > 1:
        tokens.extend((token("FECHA", 510, 600), token(dates[1], 510, 620, 70)))
    # The narrow TIPO IVA heading is stacked across two lines in the PDF.
    tokens.extend(token(char, 705 + i * 8, 335) for i, char in enumerate("TIPO"))
    tokens.extend(token(char, 730 + i * 8, 350) for i, char in enumerate("IVA"))
    for i, rate in enumerate(rates):
        tokens.append(token(rate, 715, 390 + i * 15, 35))
    if summary:
        for text, x in (("BASE", 80), ("IMPONIBLE", 100), ("CUOTA", 275), ("IVA", 315),
                        ("TOTAL", 460), ("FACTURA", 500), ("TOTAL", 650), ("A", 690), ("PAGAR", 705)):
            tokens.append(token(text, x, 710))
        tokens.extend((token(base, 135, 750, 40), token(tax_amount, 325, 750, 35),
                       token(total, 520, 750, 45), token(total, 735, 750, 45)))
    table = TableRegion("synthetic", BoundingBox(40, 330, 970, 680), (("DESCRIPCION", "CANTIDAD", "IMP. UNIDAD", "IMPORTE", "TIPO IVA", "CUOTA IVA", "TOTAL"),))
    return (PDFPage(1, page.width, page.height, "", tables=(table,)),), (tuple(tokens),)


def t01_fixture(*, summary=True):
    page = PDFPage(1, 1000, 1000, "")
    tokens = [
        token("12345678", 60, 295, 60), token("NO", 55, 325), token("DE", 65, 325), token("FACTURA", 75, 325),
        token("01.09.2026", 230, 295, 70), token("FECHA", 235, 325),
        token("0012345", 380, 295, 50), token("CODIGO", 350, 325), token("CLIENTE", 385, 325),
        token("B12345678", 80, 170, 45),
        token("C.I.F.:A87654321", 25, 100, 10, token_id="p1-v1", vertical=True),
        token("CONCEPTO", 70, 340), token("ALBARAN", 240, 340), token("FECHA", 360, 340),
        token("T.CHEQUES", 500, 340), token("TOTAL", 650, 340),
    ]
    if summary:
        for text, x in (("COMISION", 80), ("FORFAIT", 220), ("GESTION", 275), ("GASTOS", 375),
                        ("ENVIO", 430), ("BASE", 510), ("IMPONIBLE", 545), ("IVA", 690),
                        ("TOTAL", 820), ("EUROS", 865)):
            tokens.append(token(text, x, 810))
        tokens.extend((token("100,00", 485, 850, 40), token("21,00%", 650, 850, 45),
                       token("3,15", 742, 850, 30),
                       token("124,15", 840, 850, 45)))
    table = TableRegion("synthetic", BoundingBox(40, 330, 960, 750), (("CONCEPTO", "ALBARAN", "FECHA", "T.CHEQUES", "TOTAL"),))
    return (PDFPage(1, page.width, page.height, "", tables=(table,)),), (tuple(tokens),)


def t02_line_fixture(detail_tokens):
    page = PDFPage(1, 1000, 1000, "")
    header = tuple(token(text, x, 300) for text, x in (
        ("DESCRIPCION", 40), ("CANTIDAD", 390), ("IMP.", 510), ("UNIDAD", 540),
        ("IMPORTE", 640), ("TIPO IVA", 730), ("CUOTA IVA", 810), ("TOTAL", 920),
    ))
    return page, header + tuple(token(text, x, y) for text, x, y in detail_tokens)


def by_name(fields):
    return {field.name: field for field in fields}


class TemplateExtractionTests(unittest.TestCase):
    def test_offline_t02_lines_preserve_columns_sign_and_field_evidence(self):
        page = PDFPage(1, 1000, 1000, "")
        tokens = tuple(token(text, x, y) for text, x, y in (
            ("DESCRIPCION", 40, 300), ("CANTIDAD", 390, 300), ("IMP.", 510, 300),
            ("UNIDAD", 540, 300), ("IMPORTE", 640, 300), ("TIPO IVA", 730, 300),
            ("CUOTA IVA", 810, 300), ("TOTAL", 920, 300),
            ("WALLET", 40, 350), ("COMIDA", 40, 363), ("2,00", 430, 376),
            ("-5,00", 550, 376), ("-10,00", 660, 376), ("21,00", 745, 376),
            ("-2,10", 825, 376), ("-12,10", 930, 376),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].description, "WALLET COMIDA")
        self.assertEqual(lines[0].values["row_type"], "charge")
        self.assertEqual(lines[0].values["unit_price"], Decimal("-5.00"))
        self.assertEqual(lines[0].values["amount"], Decimal("-10.00"))
        self.assertEqual(lines[0].values["total"], Decimal("-12.10"))
        self.assertTrue(all(item.bounds and item.source_token_ids for item in lines[0].evidence))

    def test_offline_detail_abstains_for_multipage_and_unknown_templates(self):
        page = PDFPage(1, 1000, 1000, "")
        self.assertEqual(extract_template_lines((page, page), ((), ()), "T02")[1],
                         ("multipage-detail-not-supported",))
        self.assertEqual(extract_template_lines((page,), ((),), "unknown")[1],
                         ("unsupported-template",))

    def test_description_column_includes_text_before_quantity_header(self):
        page, tokens = t02_line_fixture((
            ("WALLET", 40, 350), ("TRANSPORTE", 100, 350), ("AGOSTO/2026", 300, 350),
            ("2,00", 430, 350), ("6,00", 550, 350), ("12,00", 660, 350),
            ("21,00", 745, 350), ("2,52", 825, 350), ("14,52", 930, 350),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        self.assertEqual(lines[0].description, "WALLET TRANSPORTE AGOSTO/2026")

    def test_multiline_description_survives_nonvalued_continuation_rows(self):
        page, tokens = t02_line_fixture((
            ("WALLET", 40, 350), ("COMIDA", 40, 362),
            ("2,00", 430, 374), ("4,00", 550, 374), ("8,00", 660, 374),
            ("21,00", 745, 374), ("1,68", 825, 374), ("9,68", 930, 374),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].description, "WALLET COMIDA")

    def test_unpriced_section_heading_does_not_prefix_the_next_charge(self):
        page, tokens = t02_line_fixture((
            ("SERVICIO", 40, 350), ("RETRIPLUS", 100, 350),
            ("UPRETRIPLUS", 40, 362), ("PLAN", 120, 362), ("JULIO/2026", 160, 362),
            ("1,00", 430, 362), ("-40,00€", 550, 362), ("-40,00", 660, 362),
            ("21,00", 745, 362), ("-8,40", 825, 362), ("-48,40", 930, 362),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        charges = [line for line in lines if not line.values.get("document_note")]
        notes = [line for line in lines if line.values.get("document_note")]
        self.assertEqual(len(charges), 1)
        self.assertEqual(charges[0].description, "UPRETRIPLUS PLAN JULIO/2026")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].values["row_type"], "informational")
        self.assertTrue(all(ev.bounds and ev.source_token_ids for ev in notes[0].evidence))

    def test_cancellation_note_near_charge_does_not_change_its_classification(self):
        page, tokens = t02_line_fixture((
            ("WALLET", 40, 350), ("COMIDA", 100, 350), ("1,00", 430, 350),
            ("-4,00€", 550, 350), ("-4,00", 660, 350), ("21,00", 745, 350),
            ("-0,84", 825, 350), ("-4,84", 930, 350),
            ("****", 40, 374), ("ANULA", 70, 374), ("FACTURA", 110, 374), ("12345678", 180, 374), ("****", 250, 374),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        charge = next(line for line in lines if not line.values.get("document_note"))
        note = next(line for line in lines if line.values.get("document_note"))
        self.assertEqual(charge.values["row_type"], "charge")
        self.assertEqual(note.values["row_type"], "cancellation")
        self.assertTrue(note.warnings)

    def test_independent_cancellation_note_is_not_emitted_as_a_charge(self):
        page, tokens = t02_line_fixture((
            ("****", 40, 350), ("ANULA", 70, 350), ("FACTURA", 110, 350),
            ("12345678", 180, 350), ("****", 250, 350),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].values["document_note"])
        self.assertEqual(lines[0].values["row_type"], "cancellation")
        self.assertTrue(all(e.bounds and e.source_token_ids for e in lines[0].evidence))

    def test_informational_beneficiary_row_preserves_quantity_only(self):
        page, tokens = t02_line_fixture((
            ("NUMERO", 40, 350), ("DE", 100, 350), ("BENEFICIARIOS", 120, 350), ("15,00", 430, 350),
        ))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].values["row_type"], "informational")
        self.assertEqual(lines[0].values["quantity"], Decimal("15.00"))
        self.assertNotIn("amount", lines[0].values)
        self.assertTrue(any(ev.quote == "15,00" for ev in lines[0].evidence))

    def test_separate_discounts_and_near_duplicate_charges_remain_independent(self):
        detail = []
        for y, description, amount in ((350, "WALLET COMIDA", "10,00"),
                                       (362, "WALLET COMIDA", "5,00"),
                                       (374, "DESCUENTO 50,00%", "-5,00"),
                                       (386, "DESCUENTO 20,00%", "-1,00")):
            detail.extend(((description, 40, y), ("1,00", 430, y), (amount, 660, y), ("21,00", 745, y),
                           ("1,05", 825, y), ("11,05", 930, y)))
        page, tokens = t02_line_fixture(tuple(detail))
        lines, warnings = extract_template_lines((page,), (tokens,), "T02")
        self.assertEqual(warnings, ())
        self.assertEqual(len(lines), 4)
        self.assertEqual([line.values["row_type"] for line in lines],
                         ["charge", "charge", "discount", "discount"])

    def test_t02_header_summary_dates_and_role_ids_have_precise_evidence(self):
        pages, tokens = t02_fixture(dates=("03.09.2026", "04.09.2026"))
        fields, warnings = extract_t02_header_summary(pages, tokens)
        values = by_name(fields)
        self.assertEqual(values["invoice_number"].value, "12345678")
        self.assertEqual(values["issue_date"].value, date(2026, 9, 3))
        self.assertEqual(values["client_code"].value, "0001234")
        self.assertEqual(values["issuer_tax_id"].value, "A87654321")
        self.assertEqual(values["receiver_tax_id"].value, "B12345678")
        self.assertEqual(values["tax_base"].value, Decimal("100.00"))
        self.assertEqual(values["tax_amount"].value, Decimal("21.00"))
        invoice_evidence = values["invoice_number"].evidence[0]
        self.assertEqual(invoice_evidence.quote, "12345678")
        self.assertEqual(invoice_evidence.method, "header-label-same-column")
        self.assertLess(invoice_evidence.bounds.x1 - invoice_evidence.bounds.x0, 100)
        self.assertEqual(invoice_evidence.extractor_id, "as400-t02-header-summary")
        self.assertTrue(warnings == ())

    def test_t02_summary_uses_labels_to_separate_close_totals(self):
        pages, tokens = t02_fixture()
        values = by_name(extract_t02_header_summary(pages, tokens)[0])
        self.assertEqual(values["total_invoice"].value, Decimal("121.00"))
        self.assertEqual(values["total_due"].value, Decimal("121.00"))
        self.assertNotEqual(values["total_invoice"].evidence[0].label_bounds,
                            values["total_due"].evidence[0].label_bounds)

    def test_t02_keeps_multiple_vat_rates_and_signed_values(self):
        pages, tokens = t02_fixture(rates=("21,00", "10,00"))
        values = by_name(extract_t02_header_summary(pages, tokens)[0])
        self.assertEqual(values["tax_rates"].value, (Decimal("10.00"), Decimal("21.00")))
        self.assertEqual(len(values["tax_rates"].evidence), 2)
        negative_pages, negative_tokens = t02_fixture(base="-40,00", tax_amount="-8,40",
                                                       total="-48,40", rates=("21,00",))
        negative = by_name(extract_t02_header_summary(negative_pages, negative_tokens)[0])
        self.assertEqual(negative["tax_base"].value, Decimal("-40.00"))
        self.assertEqual(negative["tax_amount"].value, Decimal("-8.40"))

    def test_t01_preserves_euros_candidate_without_financial_total_assignment(self):
        pages, tokens = t01_fixture()
        values = by_name(extract_t01_header_summary(pages, tokens)[0])
        self.assertEqual(values["currency"].value, "EUR")
        self.assertEqual(values["tax_base"].value, Decimal("100.00"))
        self.assertEqual(values["tax_rates"].value, (Decimal("21.00"),))
        self.assertEqual(values["tax_amount"].value, Decimal("3.15"))
        self.assertEqual(values["displayed_total_euros"].value, Decimal("124.15"))
        self.assertEqual(values["displayed_total_euros"].status, "unverified_semantics")
        self.assertEqual(values["total_invoice"].status, "ambiguous")
        self.assertIsNone(values["total_invoice"].value)
        self.assertEqual(values["total_due"].status, "ambiguous")

    def test_missing_values_and_duplicate_labels_are_not_confirmed(self):
        pages, tokens = t02_fixture(summary=False, duplicate_date=True)
        fields = by_name(extract_t02_header_summary(pages, tokens)[0])
        self.assertEqual(fields["issue_date"].status, "ambiguous")
        self.assertEqual(fields["tax_base"].status, "absent")

    def test_unreadable_value_is_not_returned_as_a_success(self):
        pages, tokens = t02_fixture(summary=False)
        bad = tuple(item for item in tokens[0] if item.text != "03.09.2026")
        bad += (token("99.99.2026", 230, 295, 70),)
        fields = by_name(extract_t02_header_summary(pages, (bad,))[0])
        self.assertEqual(fields["issue_date"].status, "unreadable")
        self.assertIsNone(fields["issue_date"].value)

    def test_wrong_column_geometry_does_not_assign_an_adjacent_amount(self):
        pages, tokens = t02_fixture()
        tokens = tuple(item for item in tokens[0] if item.text != "100,00") + (token("100,00", 800, 750, 40),)
        fields = by_name(extract_t02_header_summary(pages, (tokens,))[0])
        self.assertEqual(fields["tax_base"].status, "insufficient_evidence")
        self.assertIsNone(fields["tax_base"].value)

    def test_unknown_classification_does_not_fall_back_to_generic_or_template(self):
        page = PDFPage(1, 595, 842, "not a recognized invoice", blocks=())
        with patch("application.template_extraction.extract_pdf", return_value=(page,)), \
             patch("application.template_extraction.extract_pdf_tokens", return_value=((),)):
            result = extract_as400_header_summary(b"synthetic")
        self.assertEqual(result.classification.template, "unknown")
        self.assertIsNone(result.extractor_id)
        self.assertEqual(result.fields, ())

    def test_template_extractors_are_versioned_but_not_active(self):
        self.assertEqual(tuple(item.id for item in EXTRACTORS), ("generic-invoice",))
        self.assertEqual(tuple(item.id for item in TEMPLATE_EXTRACTORS),
                         ("as400-t01-header-summary", "as400-t02-header-summary"))
