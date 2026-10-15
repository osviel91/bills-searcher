import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import pdf_text_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]
EXISTING_PDF = ROOT / "evaluation/fixtures/synthetic-basic-001.pdf"


def synthetic_pdf(pages):
    """Build a tiny WinAnsi PDF for parser tests without a PDF-generation dependency."""
    page_count = len(pages)
    font_id = 3 + page_count * 2
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(page_count))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    for index, lines in enumerate(pages):
        page_id, stream_id = 3 + index * 2, 4 + index * 2
        commands = []
        for line in lines:
            if line[0] == "rule":
                _, x1, y1, x2, y2 = line
                commands.append(f"{x1} {y1} m {x2} {y2} l S\n")
                continue
            x, y, text = line
            encoded = text.encode("cp1252").hex().upper()
            commands.append(f"BT /F1 11 Tf {x} {y} Td <{encoded}> Tj ET\n")
        stream = "".join(commands).encode("ascii")
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                        f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {stream_id} 0 R >>").encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    output.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(output)


def _manifest(root, cases):
    path = root / "manifest.json"
    path.write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
    return path


def _child_success(command, **_kwargs):
    engine = command[-2]
    output = command[-1]
    if engine in {"markitdown", "pdfplumber"}:
        Path(output).write_text("synthetic output", encoding="utf-8")
        Path(output).chmod(0o600)
    response = {"status": "success", "page_count": 1 if engine == "inspect" else None,
                "elapsed_seconds": 0.001, "rss_max_bytes_approx": 12345,
                "output_characters": 16 if engine == "pdfplumber" else None}
    return benchmark.subprocess.CompletedProcess(command, 0, json.dumps(response).encode(), b"SENSITIVE-STDERR")


class PDFTextBenchmarkTests(unittest.TestCase):
    def test_manifest_rejects_absolute_traversal_and_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("/etc/passwd.pdf", "../outside.pdf", "C:/outside.pdf", "sub/../outside.pdf"):
                with self.subTest(path=path):
                    manifest = _manifest(root, [{"case_id": "opaque", "pdf": path}])
                    with self.assertRaisesRegex(benchmark.BenchmarkInputError, "invalid_manifest"):
                        benchmark._manifest(manifest)
            manifest = root / "bad.json"
            manifest.write_text('{"version": 1, "cases": []}', encoding="utf-8")
            with self.assertRaisesRegex(benchmark.BenchmarkInputError, "invalid_manifest"):
                benchmark._manifest(manifest)

    def test_symlink_outside_authorized_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, outside = base / "pdfs", base / "outside.pdf"
            root.mkdir()
            outside.write_bytes(b"synthetic")
            (root / "escape.pdf").symlink_to(outside)
            with self.assertRaisesRegex(benchmark.BenchmarkInputError, "path_outside_authorized_root"):
                benchmark._case_path(root.resolve(), "escape.pdf")

    def test_secure_reader_refuses_symlinks_even_when_they_point_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.pdf").write_bytes(b"synthetic")
            (root / "alias.pdf").symlink_to(root / "target.pdf")
            with self.assertRaises(OSError):
                benchmark._read_pdf(root.resolve(), "alias.pdf")

    def test_gold_metrics_are_exact_and_only_return_aggregates(self):
        result = benchmark._gold_metrics("Factura F-10 Total -2,50", {
            "text": "Factura F-10 Total -2,50",
            "critical_fields": {"number": "F-10", "amount": "-2,50"},
            "section_order": ["Factura", "Total"],
        })
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["critical_fields"], {"number": True, "amount": True})
        self.assertEqual(result["section_order"]["pair_order_fraction"], 1)
        self.assertNotIn("F-10", json.dumps(result))
        self.assertEqual(benchmark._gold_metrics("x" * 10_000, {"text": "y" * 1_000})["cer_status"],
                         "skipped_edit_limit")

    def test_timeout_of_one_extractor_does_not_prevent_the_other(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "opaque.pdf").write_bytes(b"same synthetic PDF bytes")
            manifest = _manifest(root, [{"case_id": "opaque", "pdf": "opaque.pdf"}])
            observed_inputs = []

            def child(command, **kwargs):
                engine = command[-2]
                observed_inputs.append(kwargs["input"])
                if engine == "markitdown":
                    raise benchmark.subprocess.TimeoutExpired(command, kwargs["timeout"],
                                                               output=b"DO_NOT_REPORT", stderr=b"DO_NOT_REPORT")
                return _child_success(command, **kwargs)

            with patch.object(benchmark.subprocess, "run", side_effect=child):
                run_dir, metrics = benchmark.run(manifest, root, base / "out", timeout=0.01)
            case = metrics["cases"][0]
            self.assertEqual(case["extractors"]["markitdown"]["status"], "timeout")
            self.assertEqual(case["extractors"]["pdfplumber"]["status"], "success")
            self.assertEqual(observed_inputs, [b"same synthetic PDF bytes"] * 3)
            errors = (run_dir / "errors.json").read_text()
            self.assertIn("TimeoutError", errors)
            self.assertNotIn("DO_NOT_REPORT", errors)

    def test_timeout_and_worker_error_reap_the_direct_child(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            worker = base / "temporary_worker.py"
            worker.write_text(
                "import os, sys, time\n"
                "with open(sys.argv[2] + '.pid', 'w') as pid: pid.write(str(os.getpid()))\n"
                "if sys.argv[1] == 'sleep': time.sleep(30)\n"
                "else: os._exit(7)\n",
                encoding="utf-8",
            )
            output = base / "raw.txt"
            with patch.object(benchmark, "WORKER", worker):
                timeout_result = benchmark._worker("sleep", b"synthetic", output, 1)
                timeout_pid = int(Path(str(output) + ".pid").read_text())
                error_result = benchmark._worker("error", b"synthetic", output, 5)
                error_pid = int(Path(str(output) + ".pid").read_text())
            self.assertEqual(timeout_result["status"], "timeout")
            self.assertEqual(error_result["error_type"], "WorkerError")
            for pid in (timeout_pid, error_pid):
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_pdfplumber_timeout_does_not_prevent_markitdown(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "opaque.pdf").write_bytes(b"same synthetic PDF bytes")
            manifest = _manifest(root, [{"case_id": "opaque", "pdf": "opaque.pdf"}])

            def child(command, **kwargs):
                if command[-2] == "pdfplumber":
                    raise benchmark.subprocess.TimeoutExpired(command, kwargs["timeout"])
                return _child_success(command, **kwargs)

            with patch.object(benchmark.subprocess, "run", side_effect=child):
                _, metrics = benchmark.run(manifest, root, base / "out", timeout=0.01)
            extractors = metrics["cases"][0]["extractors"]
            self.assertEqual(extractors["markitdown"]["status"], "success")
            self.assertEqual(extractors["pdfplumber"]["status"], "timeout")

    def test_missing_and_corrupt_pdfs_are_sanitized_and_remaining_cases_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "corrupt.pdf").write_bytes(b"SYNTHETIC_SECRET_NOT_A_PDF")
            manifest = _manifest(root, [
                {"case_id": "missing_case", "pdf": "missing.pdf"},
                {"case_id": "corrupt_case", "pdf": "corrupt.pdf"},
            ])
            with patch.object(benchmark, "_worker", side_effect=[
                    {"status": "failed", "error_type": "PDFSyntaxError"},
                    {"status": "failed", "error_type": "PDFSyntaxError"},
                    {"status": "failed", "error_type": "PDFSyntaxError"}]) as worker:
                run_dir, metrics = benchmark.run(manifest, root, base / "out")
            self.assertEqual(worker.call_count, 3)
            self.assertEqual(metrics["cases"][0]["input_status"], "input_error")
            self.assertEqual(metrics["cases"][1]["extractors"]["markitdown"]["status"], "failed")
            combined = "".join(path.read_text() for path in run_dir.glob("*.json"))
            self.assertNotIn("SYNTHETIC_SECRET_NOT_A_PDF", combined)
            self.assertNotIn(str(root), combined)

    def test_only_private_raw_outputs_contain_extracted_text(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "opaque.pdf").write_bytes(b"same synthetic PDF bytes")
            manifest = _manifest(root, [{"case_id": "opaque", "pdf": "opaque.pdf"}])

            def child(command, **kwargs):
                engine = command[-2]
                output = command[-1]
                if engine in {"markitdown", "pdfplumber"}:
                    Path(output).write_text("PRIVATE_SYNTHETIC_TEXT", encoding="utf-8")
                    Path(output).chmod(0o600)
                response = {"status": "success", "page_count": 1 if engine == "inspect" else None,
                            "elapsed_seconds": 0.01, "rss_max_bytes_approx": 10}
                return benchmark.subprocess.CompletedProcess(command, 0, json.dumps(response).encode(), b"")

            with patch.object(benchmark.subprocess, "run", side_effect=child):
                run_dir, _ = benchmark.run(manifest, root, base / "out")
            reports = (run_dir / "run.json").read_text() + (run_dir / "metrics.json").read_text() + (run_dir / "errors.json").read_text()
            self.assertNotIn("PRIVATE_SYNTHETIC_TEXT", reports)
            self.assertIn("PRIVATE_SYNTHETIC_TEXT", (run_dir / "raw/pdfplumber/opaque.txt").read_text())
            self.assertNotIn("input_sha256", (run_dir / "metrics.json").read_text())
            self.assertEqual(run_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual((run_dir / "metrics.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((run_dir / "raw/pdfplumber/opaque.txt").stat().st_mode & 0o777, 0o600)

    def test_empty_outputs_are_reported_without_quality_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "opaque.pdf").write_bytes(b"synthetic")
            manifest = _manifest(root, [{"case_id": "opaque", "pdf": "opaque.pdf"}])

            def child(command, **_kwargs):
                engine, output = command[-2:]
                if engine != "inspect":
                    Path(output).write_text("", encoding="utf-8")
                response = {"status": "success", "page_count": 1 if engine == "inspect" else None,
                            "elapsed_seconds": 0.001, "rss_max_bytes_approx": 123, "output_characters": 0}
                return benchmark.subprocess.CompletedProcess(command, 0, json.dumps(response).encode(), b"")

            with patch.object(benchmark.subprocess, "run", side_effect=child):
                _, metrics = benchmark.run(manifest, root, base / "out")
            for engine in ("markitdown", "pdfplumber"):
                result = metrics["cases"][0]["extractors"][engine]
                self.assertEqual(result["status"], "empty")
                self.assertEqual(result["output_characters"], 0)
                self.assertNotIn("gold", result)

    def test_pdfplumber_failure_does_not_prevent_markitdown(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "opaque.pdf").write_bytes(b"synthetic")
            manifest = _manifest(root, [{"case_id": "opaque", "pdf": "opaque.pdf"}])

            def child(command, **kwargs):
                engine = command[-2]
                if engine == "pdfplumber":
                    response = {"status": "failed", "error_type": "SyntheticParserError"}
                    return benchmark.subprocess.CompletedProcess(command, 0, json.dumps(response).encode(), b"")
                return _child_success(command, **kwargs)

            with patch.object(benchmark.subprocess, "run", side_effect=child):
                run_dir, metrics = benchmark.run(manifest, root, base / "out")
            extractors = metrics["cases"][0]["extractors"]
            self.assertEqual(extractors["markitdown"]["status"], "success")
            self.assertEqual(extractors["pdfplumber"]["status"], "failed")
            errors = json.loads((run_dir / "errors.json").read_text())
            self.assertEqual(errors["errors"][0]["error_type"], "SyntheticParserError")

    def test_markitdown_failure_does_not_prevent_pdfplumber(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            root.mkdir()
            (root / "opaque.pdf").write_bytes(b"synthetic")
            manifest = _manifest(root, [{"case_id": "opaque", "pdf": "opaque.pdf"}])

            def child(command, **kwargs):
                if command[-2] == "markitdown":
                    response = {"status": "failed", "error_type": "SyntheticParserError"}
                    return benchmark.subprocess.CompletedProcess(command, 0, json.dumps(response).encode(), b"")
                return _child_success(command, **kwargs)

            with patch.object(benchmark.subprocess, "run", side_effect=child):
                _, metrics = benchmark.run(manifest, root, base / "out")
            extractors = metrics["cases"][0]["extractors"]
            self.assertEqual(extractors["markitdown"]["status"], "failed")
            self.assertEqual(extractors["pdfplumber"]["status"], "success")

    @unittest.skipUnless(importlib.util.find_spec("markitdown") and importlib.util.find_spec("pdfplumber"),
                         "PDF parser dependencies are installed in the app test image")
    def test_actual_extractors_run_on_existing_and_generated_synthetic_pdfs(self):
        two_page = synthetic_pdf([
            [(50, 740, "FACTURA SINTETICA"), (50, 700, "ARTICULO"), (300, 700, "IMPORTE"),
             (50, 680, "Cafe"), (300, 680, "10,00 EUR"), (50, 660, "Nino"), (300, 660, "-2,50 EUR")],
            [(50, 740, "DESCRIPCION | CANTIDAD | IMPORTE"), (50, 710, "Caf\u00e9 | 1 | -2,50"),
             (50, 680, "IVA 21% | -0,53"), (50, 650, "TOTAL | -3,03"),
             ("rule", 45, 725, 560, 725), ("rule", 45, 695, 560, 695),
             ("rule", 45, 665, 560, 665), ("rule", 45, 635, 560, 635),
             ("rule", 45, 725, 45, 635), ("rule", 300, 725, 300, 635),
             ("rule", 430, 725, 430, 635), ("rule", 560, 725, 560, 635)],
        ])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "pdfs"
            (root / "fixtures").mkdir(parents=True)
            (root / "fixtures/synthetic-basic-001.pdf").write_bytes(EXISTING_PDF.read_bytes())
            (root / "fixtures/synthetic-layout-001.pdf").write_bytes(two_page)
            manifest = _manifest(root, [
                {"case_id": "synthetic_basic", "pdf": "fixtures/synthetic-basic-001.pdf",
                 "gold": {"text": "Factura F-1001 Total 12,10", "critical_fields": {"invoice": "F-1001"},
                          "section_order": ["Factura", "Total"]}},
                {"case_id": "synthetic_layout", "pdf": "fixtures/synthetic-layout-001.pdf"},
                {"case_id": "synthetic_corrupt", "pdf": "fixtures/synthetic-corrupt-001.pdf"},
            ])
            (root / "fixtures/synthetic-corrupt-001.pdf").write_bytes(b"%PDF-1.4\nSYNTHETIC_CORRUPT_PDF")
            run_dir, metrics = benchmark.run(manifest, root, base / "out", timeout=30)
            self.assertEqual(metrics["case_count"], 3)
            self.assertEqual(metrics["cases"][0]["extractors"]["markitdown"]["status"], "success")
            self.assertEqual(metrics["cases"][0]["extractors"]["pdfplumber"]["status"], "success")
            self.assertEqual(metrics["cases"][0]["extractors"]["markitdown"]["gold"]["critical_fields"]["invoice"], True)
            self.assertEqual(metrics["cases"][1]["page_count"], 2)
            self.assertEqual(metrics["cases"][1]["extractors"]["markitdown"]["status"], "success")
            self.assertEqual(metrics["cases"][1]["extractors"]["pdfplumber"]["status"], "success")
            self.assertEqual(metrics["cases"][2]["extractors"]["pdfplumber"]["status"], "failed")
            self.assertEqual(metrics["cases"][2]["page_count"], None)
            self.assertIn(metrics["cases"][2]["extractors"]["markitdown"]["status"],
                          {"success", "empty", "failed"})
            self.assertNotIn("SYNTHETIC_CORRUPT_PDF", (run_dir / "errors.json").read_text())
            self.assertTrue((run_dir / "raw/markitdown/synthetic_layout.md").is_file())
            self.assertTrue((run_dir / "raw/pdfplumber/synthetic_layout.txt").is_file())


if __name__ == "__main__":
    unittest.main()
