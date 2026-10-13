import unittest

from application.extractor_registry import TEMPLATE_DETECTORS
from application.template_detection import detect_template
from domain.structured_extraction import BoundingBox, PDFPage, TableRegion


class TemplateDetectionTests(unittest.TestCase):
    def page(self, text, tables, width=595, height=842):
        return PDFPage(1, width, height, text, tables=tuple(tables))

    def table(self, rows, top, bottom):
        return TableRegion("synthetic", BoundingBox(30, top, 575, bottom), tuple(rows))

    def test_synthetic_t01_requires_header_and_independent_support(self):
        page = self.page("COMISION FORFAIT GESTION GASTOS ENVIO TOTAL EUROS", [
            self.table([TEMPLATE_DETECTORS[0].headers], 280, 305),
            self.table([("synthetic detail",)], 310, 630),
        ])
        result = detect_template((page,))
        self.assertEqual(result.template, "T01")
        self.assertEqual(result.detector_id, "as400-t01")
        self.assertIsNone(result.reason)

    def test_synthetic_t02_uses_ordered_columns_and_summary_labels(self):
        page = self.page("BASE IMPONIBLE CUOTA IVA TOTAL FACTURA TOTAL A PAGAR", [
            self.table([TEMPLATE_DETECTORS[1].headers], 280, 572),
        ])
        result = detect_template((page,))
        self.assertEqual(result.template, "T02")
        self.assertEqual(result.signals["as400-t02"]["column_header"], ("matched",))

    def test_legacy_lookalike_without_known_order_abstains(self):
        page = self.page("COMISION FORFAIT GESTION GASTOS ENVIO TOTAL EUROS", [
            self.table([("CONCEPTO", "FECHA", "CANTIDAD", "TOTAL")], 280, 305),
            self.table([("example",)], 310, 630),
        ])
        result = detect_template((page,))
        self.assertEqual(result.template, "unknown")
        self.assertEqual(result.reason, "insufficient-structural-signals")

    def test_synthetic_conflicting_detectors_abstain_as_ambiguous(self):
        page = self.page("COMISION IVA BASE IMPONIBLE TOTAL A PAGAR", [
            self.table([("X", "Y")], 280, 572),
        ])
        detectors = (
            TEMPLATE_DETECTORS[0].__class__("as400-a", "1", ("X", "Y"), ("COMISION", "IVA"), "single-detail-band"),
            TEMPLATE_DETECTORS[1].__class__("as400-b", "1", ("X", "Y"),
                                             ("BASE IMPONIBLE", "TOTAL A PAGAR"), "single-detail-band"),
        )
        result = detect_template((page,), detectors)
        self.assertEqual(result.template, "unknown")
        self.assertEqual(result.reason, "ambiguous-template-match")

    def test_unsupported_multipage_layout_abstains(self):
        page = self.page("", [])
        result = detect_template((page, page))
        self.assertEqual(result.template, "unknown")
        self.assertEqual(result.reason, "unsupported-page-count")

    def test_header_without_a_second_signal_abstains(self):
        page = self.page("", [self.table([TEMPLATE_DETECTORS[0].headers], 400, 450)])
        result = detect_template((page,))
        self.assertEqual(result.template, "unknown")
        self.assertEqual(result.reason, "insufficient-structural-signals")


if __name__ == "__main__":
    unittest.main()
