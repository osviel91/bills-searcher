import json
import tempfile
import unittest
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import jobstore
from application.structured_extraction import extract_document
from domain.structured_extraction import (
    Evidence,
    ExtractedField,
    ExtractionExecution,
    ProposedLine,
)
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


if __name__ == "__main__":
    unittest.main()
