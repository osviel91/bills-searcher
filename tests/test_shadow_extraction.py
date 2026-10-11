import time
import unittest
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from application import shadow_extraction as shadow
from domain.structured_extraction import ExtractedField, ProposedLine


def _sleep_child(send):
    time.sleep(10)


def _send_classification_then_sleep(send):
    send.send({"template": "T02", "detector_version": "as400-t02@1",
               "extractor_versions": ("as400-t02-header-summary@1", "as400-t02-lines@1")})
    time.sleep(10)


class ShadowExtractionTests(unittest.TestCase):
    def test_disabled_mode_does_not_check_attempt_or_spawn_detector(self):
        with patch.dict("os.environ", {"AS400_SHADOW_ENABLED": "false"}), \
             patch.object(shadow, "_still_current") as current, \
             patch.object(shadow, "_run_isolated") as isolated:
            self.assertIsNone(shadow.run("opaque-doc", "attempt-1", (), b"pdf", (), ()))
        current.assert_not_called()
        isolated.assert_not_called()

    def test_enabled_mode_logs_only_safe_child_diagnostic(self):
        diagnostic = {"status": "succeeded", "template": "T01", "fields_produced": 2,
                      "lines_produced": 1, "warnings": 0}
        with patch.dict("os.environ", {"AS400_SHADOW_ENABLED": "true"}), \
             patch.object(shadow, "_still_current", return_value=True), \
             patch.object(shadow, "_run_isolated", return_value=diagnostic), \
             patch.object(shadow.LOGGER, "info") as log:
            result = shadow.run("opaque-doc", "attempt-1", (), b"pdf", (), ())
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["document_id"], "opaque-doc")
        self.assertEqual(result["attempt_id"], "attempt-1")
        self.assertIn("duration_ms", result)
        self.assertIn("generic-invoice@1", result["extractor_versions"])
        log.assert_called_once()
        self.assertEqual(log.call_args.kwargs["extra"]["as400_shadow"], result)

    def test_stale_attempt_is_discarded_without_diagnostic(self):
        with patch.dict("os.environ", {"AS400_SHADOW_ENABLED": "true"}), \
             patch.object(shadow, "_still_current", side_effect=(True, False)), \
             patch.object(shadow, "_run_isolated") as isolated, \
             patch.object(shadow.LOGGER, "info") as log:
            self.assertIsNone(shadow.run("opaque-doc", "old-attempt", (), b"pdf", (), ()))
        isolated.assert_called_once()
        log.assert_not_called()

    def test_detector_classification_selects_t01_and_t02_extractors(self):
        for template in ("T01", "T02"):
            classification = SimpleNamespace(template=template, detector_id=f"as400-{template.lower()}",
                                              detector_version="1", reason=None)
            header = Mock(extract=Mock(return_value=((ExtractedField("invoice_number", "special-value"),), (), ())))
            lines = Mock(extract=Mock(return_value=((), (ProposedLine("line"),), ())))
            with self.subTest(template=template), \
                 patch.object(shadow, "detect_template", return_value=classification), \
                 patch.object(shadow, "extract_pdf_tokens", return_value=(("tokens",),)), \
                 patch.object(shadow, "template_extractor_for", return_value=header) as get_header, \
                 patch.object(shadow, "template_line_extractor_for", return_value=lines) as get_lines:
                result = shadow._diagnose(("page",), b"pdf", (), ())
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["template"], template)
            get_header.assert_called_once_with(template)
            get_lines.assert_called_once_with(template)
            header.extract.assert_called_once()
            lines.extract.assert_called_once()

    def test_unknown_and_ambiguous_classifications_abstain(self):
        for reason in ("insufficient-structural-signals", "ambiguous-template-match",
                       "unsupported-page-count"):
            classification = SimpleNamespace(template="unknown", detector_id="as400-template-router",
                                              detector_version="1", reason=reason)
            with self.subTest(reason=reason), patch.object(shadow, "detect_template", return_value=classification), \
                 patch.object(shadow, "extract_pdf_tokens") as tokens:
                result = shadow._diagnose(("pages",), b"pdf", (), ())
            self.assertEqual(result["status"], "abstained")
            self.assertEqual(result["abstention"], reason)
            tokens.assert_not_called()

    def test_detector_and_extractor_errors_are_classified_without_messages(self):
        with patch.object(shadow, "detect_template", side_effect=ValueError("customer CIF secret")):
            result = shadow._diagnose((), b"pdf", (), ())
        self.assertEqual((result["status"], result["error_type"]), ("detector_error", "ValueError"))
        self.assertNotIn("secret", repr(result))

        classification = SimpleNamespace(template="T01", detector_id="as400-t01",
                                          detector_version="1", reason=None)
        with patch.object(shadow, "detect_template", return_value=classification), \
             patch.object(shadow, "extract_pdf_tokens", side_effect=RuntimeError("amount secret")):
            result = shadow._diagnose(("page",), b"pdf", (), ())
        self.assertEqual((result["status"], result["error_type"]), ("extractor_error", "RuntimeError"))
        self.assertNotIn("amount", repr(result))

    def test_timeout_terminates_extraction_process(self):
        result = shadow._run_isolated(_sleep_child, (), 0.1)
        self.assertEqual(result, {"status": "timeout", "error_type": "TimeoutError"})

    def test_timeout_preserves_classification_and_versions_already_observed(self):
        result = shadow._run_isolated(_send_classification_then_sleep, (), 2)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["template"], "T02")
        self.assertEqual(result["detector_version"], "as400-t02@1")
        self.assertEqual(result["extractor_versions"],
                         ("as400-t02-header-summary@1", "as400-t02-lines@1"))

    def test_comparison_only_uses_equivalent_fields_and_never_values(self):
        secret = "CLIENT CIF 12345678"
        generic = (ExtractedField("invoice_number", " INV-1 "),
                   ExtractedField("customer_tax_id", " B12345678 "),
                   ExtractedField("total", Decimal("10.00")),
                   ExtractedField("issue_date", date(2026, 1, 2)))
        specialized = (ExtractedField("invoice_number", "inv-1"),
                      ExtractedField("receiver_tax_id", "b12345678"),
                      ExtractedField("total_due", Decimal("20.00")),
                      ExtractedField("issue_date", date(2026, 1, 2)))
        result = shadow.compare_fields(generic, specialized, (ProposedLine(),), ())
        self.assertEqual(result["fields_in_both"], ("invoice_number", "issue_date", "customer_tax_id"))
        self.assertEqual(result["normalization_discrepancies"], ("invoice_number", "customer_tax_id"))
        self.assertEqual(result["fields_generic_only"], ())
        self.assertEqual(result["fields_specialized_only"], ())
        self.assertEqual(result["line_count_difference"], -1)
        self.assertNotIn(secret, repr(result))

    def test_non_equivalent_financial_totals_are_not_compared(self):
        result = shadow.compare_fields(
            (ExtractedField("total", Decimal("10.00")),),
            (ExtractedField("total_invoice", Decimal("10.00")),
             ExtractedField("total_due", Decimal("10.00"))), (), (),
        )
        self.assertEqual(result["fields_in_both"], ())
        self.assertEqual(result["fields_generic_only"], ())
        self.assertEqual(result["fields_specialized_only"], ())

    def test_equivalent_decimal_values_can_report_format_normalization_difference(self):
        result = shadow.compare_fields(
            (ExtractedField("tax_base", Decimal("10.0")),),
            (ExtractedField("tax_base", Decimal("10.00")),), (), (),
        )
        self.assertEqual(result["fields_in_both"], ("tax_base",))
        self.assertEqual(result["normalization_discrepancies"], ("tax_base",))

    def test_non_equivalent_field_presence_is_reported_without_values(self):
        result = shadow.compare_fields(
            (ExtractedField("tax_base", Decimal("99.00")),),
            (ExtractedField("tax_amount", Decimal("33.00")),), (), (),
        )
        self.assertEqual(result["fields_specialized_only"], ("vat",))
        self.assertEqual(result["fields_generic_only"], ("tax_base",))
        self.assertNotIn("99.00", repr(result))
        self.assertNotIn("33.00", repr(result))


if __name__ == "__main__":
    unittest.main()
