import unittest

from invoice_metadata import extract_invoice_metadata


class InvoiceMetadataTests(unittest.TestCase):
    def test_extracts_identity_concepts_and_distinct_totals_with_evidence(self):
        text = """CHEQUE DEJEUNER ESPAÑA S.A. C.I.F.: A78887049
ACME SERVICIOS S.L.
B12345678
DIRECCION FISCAL
F123456 21.09.2026 001234 Nº DE FACTURA FECHA CODIGO CLIENTE
CONCEPTO ALBARAN FECHA CANTIDAD TOTAL
EDUCAINFANTIL 205017 21.09.2026 1 300,00
BASE IMPONIBLE CUOTA IVA TOTAL FACTURA TOTAL A PAGAR
0,00 0,00 0,00 300,00
"""
        result = extract_invoice_metadata(text)

        self.assertEqual(result["invoice_number"], "F123456")
        self.assertEqual(result["invoice_date"], "2026-09-21")
        self.assertEqual(result["customer_code"], "001234")
        self.assertEqual(result["customer_tax_id"], "B12345678")
        self.assertEqual(result["delivery_note_ids"], ["205017"])
        self.assertTrue(any("EDUCAINFANTIL" in concept for concept in result["concepts"]))
        self.assertEqual(result["tax_base"], 0)
        self.assertEqual(result["vat"], 0)
        self.assertEqual(result["invoice_total"], 0)
        self.assertEqual(result["total_payable"], 300)
        self.assertIn("total_payable", result["evidence"])
        self.assertGreaterEqual(result["confidence"]["invoice_number"], 0.8)

    def test_missing_or_ambiguous_fields_remain_unknown(self):
        result = extract_invoice_metadata("Documento sin campos de factura ni tabla de conceptos")

        self.assertIsNone(result["invoice_number"])
        self.assertIsNone(result["customer_code"])
        self.assertIsNone(result["total_payable"])
        self.assertEqual(result["concepts"], [])

    def test_extracts_fields_from_markdown_tables(self):
        text = """| B12345678 | | |
| | 25.08.2026 | 0016955 |
41603471
| Nº DE FACTURA | FECHA | CODIGO CLIENTE |
| ------------- | ----- | -------------- |
|             | CANTIDAD | TOTAL |
| DESCRIPCION | | |
WALLET COMIDA AGOSTO/2026 11,00 4,00€ 44,00
| SUBTOTAL | | |
| Base Imponible | | TOTAL FACTURA | | TOTAL A PAGAR |
| 105,04 | | 127,10 | | 127,10 |
"""
        result = extract_invoice_metadata(text)

        self.assertEqual(result["invoice_number"], "41603471")
        self.assertEqual(result["invoice_date"], "2026-08-25")
        self.assertEqual(result["customer_code"], "0016955")
        self.assertEqual(result["customer_tax_id"], "B12345678")
        self.assertTrue(result["concepts"])
        self.assertEqual(result["total_payable"], 127.1)


if __name__ == "__main__":
    unittest.main()
