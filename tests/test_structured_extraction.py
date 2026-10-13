import json
import tempfile
import unittest
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import jobstore
from application.structured_extraction import extract_document
from application.extractor_registry import Extractor, select_extractor
from domain.structured_extraction import (
    Evidence,
    ExtractedField,
    ExtractionExecution,
    ProposedLine,
)
from domain.invoice_extraction import extract_invoice
from infrastructure.extraction_store import save_execution
from infrastructure.pdf_extraction import extract_pdf


def synthetic_pdf(pages):
    font_id = 3 + len(pages) * 2
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{3 + i * 2} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>",
    ]
    for index, lines in enumerate(pages):
        content_id = 4 + index * 2
        commands = "BT /F1 12 Tf 50 750 Td " + " ".join(
            f"({line}) Tj 0 -18 Td" for line in lines
        ) + " ET"
        objects.extend([
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>",
            f"<< /Length {len(commands)} >>\nstream\n{commands}\nendstream",
        ])
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, item in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n{item}\nendobj\n".encode("ascii"))
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010} 00000 n \n".encode("ascii"))
    output.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii"))
    return bytes(output)


class StructuredExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "jobs.sqlite3"
        self.database_url = patch.object(jobstore, "DATABASE_URL", "")
        self.database_url.start()
        jobstore._initialized.clear()
        with closing(jobstore.connect(self.db_path)) as db:
            db.execute("INSERT INTO jobs(id,filename,object_key,status) VALUES(?,?,?,'indexed')",
                       ("invoice-1", "invoice.pdf", "documents/invoice.pdf"))
            db.commit()

    def tearDown(self):
        self.database_url.stop()
        self.temp.cleanup()

    def test_extracts_page_text_blocks_and_coordinates(self):
        pages = extract_pdf(synthetic_pdf([
            ["Invoice 2026-04", "Total EUR 123.45"],
            ["Second page", "No invoice number here"],
        ]))
        self.assertEqual(len(pages), 2)
        self.assertIn("Total EUR 123.45", pages[0].text)
        amount = next(block for block in pages[0].blocks if "123.45" in block.text)
        self.assertEqual(amount.id, "p1-b2")
        self.assertGreater(amount.bounds.x0, 0)
        self.assertGreater(amount.bounds.bottom, amount.bounds.top)

    def test_persists_decimal_fields_lines_evidence_and_multiple_runs(self):
        original = synthetic_pdf([["Total EUR 123.45", "Service line"]])
        with patch("application.structured_extraction.get_object",
                   return_value=type("Blob", (), {"readall": lambda self: original})()) as get_object:
            first_id = extract_document("invoice-1", db_path=self.db_path)
            second_id = extract_document("invoice-1", db_path=self.db_path)
        get_object.assert_any_call("documents/invoice.pdf")

        page = extract_pdf(original)[0]
        block = next(block for block in page.blocks if "123.45" in block.text)
        evidence = Evidence(1, block.text, block.bounds, block.id)
        save_execution(ExtractionExecution(
            id="with-evidence", document_id="invoice-1", extractor_version="test-v1",
            status="succeeded", pages=(page,),
            fields=(ExtractedField("total", Decimal("123.45"), (evidence,)),
                    ExtractedField("invoice_number", None, warnings=("absent",))),
            lines=(ProposedLine("Service line", {"amount": Decimal("123.45")}, (evidence,)),),
        ), self.db_path)

        with closing(jobstore.connect(self.db_path)) as db:
            runs = list(db.execute("SELECT id,status FROM extraction_runs ORDER BY id"))
            field = db.execute("SELECT value_json FROM extraction_fields WHERE run_id=? AND name='total'",
                               ("with-evidence",)).fetchone()
            absent = db.execute("SELECT value_json FROM extraction_fields WHERE run_id=? AND name='invoice_number'",
                                ("with-evidence",)).fetchone()
            evidence_row = db.execute("SELECT page_number,block_id,bounds_json FROM extraction_evidence "
                                      "WHERE run_id=? AND field_name='total'", ("with-evidence",)).fetchone()
            line = db.execute("SELECT values_json FROM extraction_lines WHERE run_id=?",
                              ("with-evidence",)).fetchone()
        self.assertEqual({row["id"] for row in runs}, {first_id, second_id, "with-evidence"})
        self.assertTrue(all(row["status"] == "succeeded" for row in runs))
        self.assertEqual(json.loads(field["value_json"]), "123.45")
        self.assertIsNone(json.loads(absent["value_json"]))
        self.assertEqual(evidence_row["page_number"], 1)
        self.assertEqual(evidence_row["block_id"], "p1-b1")
        self.assertEqual(json.loads(evidence_row["bounds_json"])["x0"], block.bounds.x0)
        self.assertEqual(json.loads(line["values_json"])["amount"], "123.45")
        self.assertEqual(get_object.return_value.readall(), original)

    def test_failed_pdf_extraction_is_recorded_without_changing_document_status(self):
        with patch("application.structured_extraction.get_object",
                   return_value=type("Blob", (), {"readall": lambda self: b"not a PDF"})()):
            run_id = extract_document("invoice-1", db_path=self.db_path)
        with closing(jobstore.connect(self.db_path)) as db:
            run = db.execute("SELECT status,error FROM extraction_runs WHERE id=?", (run_id,)).fetchone()
            document = db.execute("SELECT status FROM jobs WHERE id='invoice-1'").fetchone()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(document["status"], "indexed")

    def test_shadow_failure_does_not_fail_generic_extraction_or_redownload_pdf(self):
        from domain.structured_extraction import PDFPage

        data = b"synthetic pdf bytes already loaded by ingestion"
        with patch("application.structured_extraction.extract_pdf", return_value=(
                PDFPage(1, 612, 792, "Synthetic invoice"),)), \
             patch("application.structured_extraction.get_object") as get_object, \
             patch("application.shadow_extraction.run", side_effect=RuntimeError("sensitive value")):
            run_id = extract_document("invoice-1", db_path=self.db_path, data=data, attempt_id="attempt-1")

        get_object.assert_not_called()
        with closing(jobstore.connect(self.db_path)) as db:
            run = db.execute("SELECT status,extractor_version FROM extraction_runs WHERE id=?", (run_id,)).fetchone()
            document = db.execute("SELECT status FROM jobs WHERE id='invoice-1'").fetchone()
        self.assertEqual(run["status"], "succeeded")
        self.assertEqual(run["extractor_version"], "generic-invoice@1")
        self.assertEqual(document["status"], "indexed")

    def test_generic_extractor_extracts_only_evidenced_values(self):
        pdf = synthetic_pdf([["Factura: F-2026/14", "Fecha: 3/9/2026",
                              "Emisor CIF: B12345678", "Cliente CIF: A1234567B",
                              "Moneda EUR", "Descripcion  Cantidad  Precio  Importe",
                              "Servicio mensual", "Licencia    2    10,00    20,00",
                              "Descuento -5,00", "Base imponible: 15,00", "IVA 21%: 3,15",
                              "Total: 18,15"]])
        with patch("application.structured_extraction.get_object",
                   return_value=type("Blob", (), {"readall": lambda self: pdf})()):
            run_id = extract_document("invoice-1", db_path=self.db_path)
        with closing(jobstore.connect(self.db_path)) as db:
            run = db.execute("SELECT extractor_version,warnings_json FROM extraction_runs WHERE id=?", (run_id,)).fetchone()
            fields = {row["name"]: json.loads(row["value_json"]) for row in db.execute(
                "SELECT name,value_json FROM extraction_fields WHERE run_id=?", (run_id,))}
            evidence = list(db.execute("SELECT field_name,quote,page_number,bounds_json FROM extraction_evidence WHERE run_id=?", (run_id,)))
            lines = list(db.execute("SELECT description,values_json FROM extraction_lines WHERE run_id=?", (run_id,)))
        self.assertEqual(run["extractor_version"], "generic-invoice@1")
        self.assertEqual(fields["invoice_number"], "F-2026/14")
        self.assertEqual(fields["issue_date"], "2026-09-03")
        self.assertEqual(fields["tax_base"], "15.00")
        self.assertEqual(fields["vat"], "3.15")
        self.assertEqual(fields["total"], "18.15")
        self.assertIsNone(fields["customer_id"])
        self.assertTrue(lines)
        self.assertTrue(all(row["page_number"] == 1 and row["quote"] for row in evidence))
        self.assertTrue(all(row["bounds_json"] for row in evidence))

    def test_extractor_selection_reports_equal_priority_conflict(self):
        page = extract_pdf(synthetic_pdf([["Factura: F-12345"]]))
        first = Extractor("template-a", "1", 10, lambda _: True, lambda _: ((), (), ()))
        second = Extractor("template-b", "1", 10, lambda _: True, lambda _: ((), (), ()))
        selected, warnings = select_extractor(page, (first, second))
        self.assertEqual(selected.id, "generic-invoice")
        self.assertIn("template-a, template-b", warnings[0])

    def test_synthetic_date_formats_absent_fields_and_ambiguous_labels(self):
        for raw_date, expected in (("03/09/2026", "2026-09-03"), ("3-9-2026", "2026-09-03"),
                                   ("2026-09-03", "2026-09-03"), ("03.09.2026", "2026-09-03")):
            pages = extract_pdf(synthetic_pdf([[f"Fecha: {raw_date}", "Factura: Total 20,00"]]))
            fields, _, _ = extract_invoice(pages)
            values = {field.name: field.value for field in fields}
            self.assertEqual(values["issue_date"].isoformat(), expected)
            self.assertIsNone(values["invoice_number"])
            self.assertIsNone(values["tax_base"])

    def test_multiple_lines_and_multiline_description_keep_source_evidence(self):
        pages = extract_pdf(synthetic_pdf([[
            "Descripcion | Cantidad | Precio | Importe", "Servicio mensual", "soporte remoto",
            "Plan A | 2 | 10,00 | 20,00", "Plan B | 1 | 7,50 | 7,50", "Total: 27,50",
        ]]))
        _, lines, _ = extract_invoice(pages)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].description, "Servicio mensual soporte remoto Plan A")
        self.assertEqual(lines[0].values["amount"], Decimal("20.00"))
        self.assertEqual(lines[1].values["amount"], Decimal("7.50"))
        self.assertTrue(all(line.evidence for line in lines))

    def test_monetary_values_cannot_be_persisted_without_evidence(self):
        with self.assertRaisesRegex(ValueError, "requires document evidence"):
            save_execution(ExtractionExecution(
                id="no-evidence", document_id="invoice-1", extractor_version="test-v1",
                status="succeeded", fields=(ExtractedField("total", Decimal("4.00")),),
            ), self.db_path)


if __name__ == "__main__":
    unittest.main()
