import unittest
from datetime import date
from decimal import Decimal

from application.template_detection import TemplateDetection
from domain.structured_extraction import BoundingBox, Evidence, ExtractedField, ProposedLine
from infrastructure.specialized_extraction_writer import (
    MAX_SPECIALIZED_RESULT_TRANSFER_BYTES, Component, SpecializedResult, _encode,
    _payload_fingerprint, adapt_template_result, deserialize_specialized_result,
    serialize_specialized_result,
)
from test_template_extraction import t01_fixture, t02_fixture
from application.template_extraction import extract_t01_header_summary, extract_t02_header_summary


class SpecializedWriterContractTests(unittest.TestCase):
    def test_t01_result_keeps_unverified_total_and_decimal_values(self):
        classification = TemplateDetection("T01", "as400-t01", "1", {}, None)
        fields = (
            ExtractedField("displayed_total_euros", Decimal("124.15"), status="unverified_semantics"),
            ExtractedField("total_due", None, status="ambiguous", candidates=(Decimal("124.15"),)),
            ExtractedField("issue_date", date(2026, 9, 1), status="confirmed"),
        )
        result = adapt_template_result(classification, "t01-header", "1", fields,
            document_id="doc", document_sha256="a" * 64, attempt_id="attempt",
            idempotency_key="key", extractor_set_fingerprint="b" * 64)
        result.validate()
        self.assertEqual(result.fields[0].value, Decimal("124.15"))
        self.assertEqual(result.fields[1].status, "ambiguous")
        self.assertEqual(_encode(result.fields[0].value), '"124.15"')
        self.assertEqual(_encode(result.fields[2].value), '"2026-09-01"')

    def test_t02_line_values_and_multiple_geometry_evidence_are_preserved(self):
        classification = TemplateDetection("T02", "as400-t02", "1", {}, None)
        evidence = Evidence(1, "-10,00", BoundingBox(1, 2, 3, 4), "block", "IMPORTE",
                            BoundingBox(0, 1, 4, 5), "detail-column-center", "lines", "2", ("tok-1",))
        line = ProposedLine("DESCUENTO", {"amount": Decimal("-10.00"), "row_type": "discount"},
                            (evidence, evidence), ("synthetic-warning",))
        result = adapt_template_result(classification, "t02-header", "1", (), (line,),
            document_id="doc", document_sha256="a" * 64, attempt_id="attempt",
            idempotency_key="key", extractor_set_fingerprint="b" * 64,
            line_extractor=Component("t02-lines", "2"))
        self.assertEqual(result.lines[0].values["amount"], Decimal("-10.00"))
        self.assertEqual(len(result.lines[0].evidence), 2)
        self.assertEqual([item.id for item in result.components], ["t02-header", "t02-lines"])

    def test_adapter_preserves_actual_t01_and_t02_extractor_contracts(self):
        t01_pages, t01_tokens = t01_fixture()
        t01_fields, _ = extract_t01_header_summary(t01_pages, t01_tokens)
        t01 = adapt_template_result(TemplateDetection("T01", "as400-t01", "1", {}, None),
            "as400-t01-header-summary", "1", t01_fields, document_id="doc",
            document_sha256="a" * 64, attempt_id="attempt", idempotency_key="t01",
            extractor_set_fingerprint="b" * 64)
        t01_values = {field.name: field for field in t01.fields}
        self.assertEqual(t01_values["displayed_total_euros"].status, "unverified_semantics")
        self.assertEqual(t01_values["total_due"].status, "ambiguous")
        self.assertEqual(t01_values["tax_base"].value, Decimal("100.00"))

        t02_pages, t02_tokens = t02_fixture()
        t02_fields, _ = extract_t02_header_summary(t02_pages, t02_tokens)
        t02 = adapt_template_result(TemplateDetection("T02", "as400-t02", "1", {}, None),
            "as400-t02-header-summary", "1", t02_fields, document_id="doc",
            document_sha256="a" * 64, attempt_id="attempt", idempotency_key="t02",
            extractor_set_fingerprint="c" * 64)
        t02_values = {field.name: field for field in t02.fields}
        self.assertEqual(t02_values["total_due"].value, Decimal("121.00"))
        self.assertTrue(t02_values["total_due"].evidence[0].bounds)

    def test_unknown_template_is_succeeded_abstention_without_observations(self):
        classification = TemplateDetection("unknown", "router", "1", {}, "unsupported-page-count")
        result = adapt_template_result(classification, None, None, (),
            document_id="doc", document_sha256="a" * 64, attempt_id="attempt",
            idempotency_key="key", extractor_set_fingerprint="b" * 64)
        self.assertEqual((result.status, result.template, result.abstention_reason),
                         ("succeeded", "unknown", "unsupported-page-count"))
        self.assertEqual(result.fields, ())

    def test_validation_rejects_implicit_status_float_and_partial_success(self):
        common = dict(document_id="doc", document_sha256="a" * 64, attempt_id="a",
            execution_mode="specialized", detector_id="d", detector_version="1", template="T02",
            components=(Component("extractor", "1"),), extractor_set_fingerprint="b" * 64,
            result_schema_version=1, idempotency_key="key", fields=(), lines=(), status="succeeded")
        with self.assertRaisesRegex(ValueError, "explicit valid"):
            SpecializedResult(**{**common, "fields": (ExtractedField("field"),)}).validate()
        with self.assertRaisesRegex(ValueError, "binary floats"):
            SpecializedResult(**{**common, "lines": (ProposedLine(values={"amount": 1.2}),)}).validate()
        with self.assertRaisesRegex(ValueError, "partial observations"):
            SpecializedResult(**{**common, "status": "failed",
                                 "fields": (ExtractedField("field", "x", status="confirmed"),)}).validate()

    def test_result_fingerprint_is_canonical_precise_and_excludes_execution_identity(self):
        common = dict(document_id="doc", document_sha256="a" * 64, attempt_id="attempt-1",
            execution_mode="specialized", detector_id="d", detector_version="1", template="T02",
            components=(Component("extractor", "1"),), extractor_set_fingerprint="b" * 64,
            result_schema_version=1, idempotency_key="key-1", fields=(), lines=(), status="succeeded")
        amount = Decimal("123456789012345678901234567890.123456789")
        one = SpecializedResult(**{**common, "lines": (ProposedLine(values={"amount": amount,
            "quantity": Decimal("1.00")}),)})
        reordered = SpecializedResult(**{**common, "attempt_id": "attempt-2", "idempotency_key": "key-2",
            "lines": (ProposedLine(values={"quantity": Decimal("1.00"), "amount": amount}),),
            "run_id": "random-run-id"})
        self.assertEqual(_payload_fingerprint(one), _payload_fingerprint(reordered))
        self.assertEqual(_encode(amount), '"123456789012345678901234567890.123456789"')

    def test_result_fingerprint_covers_status_candidates_and_evidence(self):
        common = dict(document_id="doc", document_sha256="a" * 64, attempt_id="attempt",
            execution_mode="specialized", detector_id="d", detector_version="1", template="T02",
            components=(Component("extractor", "1"),), extractor_set_fingerprint="b" * 64,
            result_schema_version=1, idempotency_key="key", lines=(), status="succeeded")
        evidence = Evidence(1, "12,30", BoundingBox(1, 2, 3, 4), "block-a", source_token_ids=("token-a",))
        baseline = SpecializedResult(**{**common, "fields": (
            ExtractedField("total_due", Decimal("12.30"), evidence=(evidence,), status="ambiguous",
                           candidates=(Decimal("12.30"), Decimal("13.20"))),)})
        status_changed = SpecializedResult(**{**common, "fields": (
            ExtractedField("total_due", Decimal("12.30"), evidence=(evidence,), status="confirmed",
                           candidates=(Decimal("12.30"), Decimal("13.20"))),)})
        evidence_changed = SpecializedResult(**{**common, "fields": (
            ExtractedField("total_due", Decimal("12.30"), evidence=(Evidence(1, "12,30",
                           BoundingBox(1, 2, 3, 4), "block-b", source_token_ids=("token-a",)),),
                           status="ambiguous", candidates=(Decimal("12.30"), Decimal("13.20"))),)})
        candidates_changed = SpecializedResult(**{**common, "fields": (
            ExtractedField("total_due", Decimal("12.30"), evidence=(evidence,), status="ambiguous",
                           candidates=(Decimal("12.30"), Decimal("99.99"))),)})
        self.assertNotEqual(_payload_fingerprint(baseline), _payload_fingerprint(status_changed))
        self.assertNotEqual(_payload_fingerprint(baseline), _payload_fingerprint(evidence_changed))
        self.assertNotEqual(_payload_fingerprint(baseline), _payload_fingerprint(candidates_changed))

    def test_adapted_idempotency_key_survives_process_restart_and_attempt_change(self):
        classification = TemplateDetection("T02", "router", "1", {}, None)
        arguments = dict(document_id="doc", document_sha256="a" * 64, extractor_set_fingerprint="b" * 64)
        fields = (ExtractedField("total_due", Decimal("12.30"), status="confirmed"),)
        first = adapt_template_result(classification, "header", "1", fields,
            attempt_id="attempt-one", **arguments)
        restarted = adapt_template_result(classification, "header", "1", fields,
            attempt_id="attempt-two", **arguments)
        intentional_replay = adapt_template_result(classification, "header", "1", fields,
            attempt_id="attempt-two", operation_id="rerun-2", **arguments)
        self.assertEqual(first.idempotency_key, restarted.idempotency_key)
        self.assertNotEqual(first.run_id, restarted.run_id)
        self.assertNotEqual(first.idempotency_key, intentional_replay.idempotency_key)

    def test_specialized_result_json_transfer_roundtrips_decimal_and_evidence_with_a_hard_limit(self):
        evidence = Evidence(2, "-12,30", BoundingBox(1, 2, 3, 4), "block-2",
                            method="table-cell", extractor_id="lines", extractor_version="1",
                            source_token_ids=("token-1",))
        result = SpecializedResult(
            "doc", "a" * 64, "attempt", "shadow-specialized", "router", "1", "T02",
            (Component("header", "1"), Component("lines", "1")), "b" * 64, 1, "stable-key",
            (ExtractedField("amount", Decimal("-12.30"), evidence=(evidence,),
                            status="confirmed", candidates=(Decimal("-12.30"),)),),
            (ProposedLine("synthetic", {"amount": Decimal("-12.30"), "row_type": "discount"},
                          (evidence,)),), "succeeded",
        ).validate()
        decoded = deserialize_specialized_result(serialize_specialized_result(result))
        self.assertEqual(decoded.idempotency_key, result.idempotency_key)
        self.assertEqual(decoded.attempt_id, result.attempt_id)
        self.assertEqual(decoded.fields[0].value, Decimal("-12.30"))
        self.assertEqual(decoded.fields[0].evidence[0].page_number, 2)
        self.assertEqual(decoded.lines[0].values["row_type"], "discount")
        with self.assertRaisesRegex(ValueError, "transfer shape"):
            deserialize_specialized_result(b'{"unexpected":true}')
        with self.assertRaisesRegex(ValueError, "transfer size limit"):
            serialize_specialized_result(result, max_bytes=1)
        with self.assertRaisesRegex(ValueError, "transfer exceeds the size limit"):
            deserialize_specialized_result(b" " * (MAX_SPECIALIZED_RESULT_TRANSFER_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
