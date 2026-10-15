#!/usr/bin/env python3
"""Run a local, isolated comparison of MarkItDown and pdfplumber on PDFs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import stat
import subprocess
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CASES = 500
MAX_GOLD_TEXT = 20_000
MAX_GOLD_FIELDS = 100
MAX_GOLD_SECTIONS = 100
MAX_EDIT_CELLS = 10_000_000
CASE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
TOKEN = re.compile(r"\w+(?:[.,/-]\w+)*|[^\w\s]", re.UNICODE)
NUMBER = re.compile(r"(?<![\w])[-+−]?\d+(?:[.,]\d+)*(?![\w])", re.UNICODE)
WORKER = Path(__file__).with_name("pdf_text_benchmark_worker.py")


class BenchmarkInputError(ValueError):
    """A sanitized manifest or path validation error."""


def _manifest(path):
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise BenchmarkInputError("invalid_manifest")
        value = json.loads(raw)
    except BenchmarkInputError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkInputError("invalid_manifest") from None
    if not isinstance(value, dict) or value.get("version") != 1:
        raise BenchmarkInputError("invalid_manifest")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise BenchmarkInputError("invalid_manifest")
    seen = set()
    checked = []
    for case in cases:
        if not isinstance(case, dict) or set(case) - {"case_id", "pdf", "gold"}:
            raise BenchmarkInputError("invalid_manifest")
        case_id, name = case.get("case_id"), case.get("pdf")
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id) or case_id in seen:
            raise BenchmarkInputError("invalid_manifest")
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise BenchmarkInputError("invalid_manifest")
        relative = Path(name)
        windows_path = PureWindowsPath(name)
        if (relative.is_absolute() or windows_path.is_absolute() or windows_path.drive or
                ".." in relative.parts):
            raise BenchmarkInputError("invalid_manifest")
        if relative.suffix.lower() != ".pdf":
            raise BenchmarkInputError("invalid_manifest")
        gold = case.get("gold")
        if gold is not None:
            if not isinstance(gold, dict) or set(gold) - {"text", "critical_fields", "section_order"}:
                raise BenchmarkInputError("invalid_manifest")
            if "text" in gold and (not isinstance(gold["text"], str) or len(gold["text"]) > MAX_GOLD_TEXT):
                raise BenchmarkInputError("invalid_manifest")
            fields = gold.get("critical_fields", {})
            order = gold.get("section_order", [])
            if (not isinstance(fields, dict) or len(fields) > MAX_GOLD_FIELDS or
                    any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                                                     for k, v in fields.items()) or
                    not isinstance(order, list) or len(order) > MAX_GOLD_SECTIONS or
                    any(not isinstance(item, str) or not item.strip() for item in order)):
                raise BenchmarkInputError("invalid_manifest")
        seen.add(case_id)
        checked.append({"case_id": case_id, "pdf": name, "gold": gold})
    return checked


def _case_path(root, name):
    try:
        candidate = (root / name).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise BenchmarkInputError("invalid_pdf_path") from None
    if not candidate.is_relative_to(root) or candidate == root:
        raise BenchmarkInputError("path_outside_authorized_root")
    return candidate


def _read_pdf(root, name):
    """Open each path component without following symlinks or leaving root."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("secure path opening is unavailable")
    parts = Path(name).parts
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    directory_flags = flags | os.O_DIRECTORY
    current = os.open(root, directory_flags)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = next_fd
        descriptor = os.open(parts[-1], flags, dir_fd=current)
    finally:
        os.close(current)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("not a regular file")
        with os.fdopen(descriptor, "rb") as source:
            return source.read()
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _levenshtein(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def _gold_metrics(text, gold):
    result = {}
    if "text" in gold:
        expected = unicodedata.normalize("NFC", gold["text"])
        actual = unicodedata.normalize("NFC", text)
        if (len(expected) + 1) * (len(actual) + 1) <= MAX_EDIT_CELLS:
            result["cer"] = _levenshtein(expected, actual) / max(1, len(expected))
            result["cer_status"] = "computed"
        else:
            result["cer"], result["cer_status"] = None, "skipped_edit_limit"
        expected_words, actual_words = TOKEN.findall(expected), TOKEN.findall(actual)
        if (len(expected_words) + 1) * (len(actual_words) + 1) <= MAX_EDIT_CELLS:
            result["wer"] = _levenshtein(expected_words, actual_words) / max(1, len(expected_words))
            result["wer_status"] = "computed"
        else:
            result["wer"], result["wer_status"] = None, "skipped_edit_limit"
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if gold.get("critical_fields"):
        result["critical_fields"] = {name: " ".join(unicodedata.normalize("NFKC", value).casefold().split()) in normalized
                                     for name, value in gold["critical_fields"].items()}
    if gold.get("section_order"):
        expected_sections = gold["section_order"]
        positions = [normalized.find(" ".join(unicodedata.normalize("NFKC", item).casefold().split()))
                     for item in expected_sections]
        pairs = [(i, j) for i in range(len(positions)) for j in range(i + 1, len(positions))
                 if positions[i] >= 0 and positions[j] >= 0]
        result["section_order"] = {"sections_expected": len(expected_sections),
                                   "sections_found": sum(position >= 0 for position in positions),
                                   "pairs_observed": len(pairs),
                                   "pairs_in_expected_order": sum(positions[i] < positions[j] for i, j in pairs),
                                   "pair_order_fraction": (sum(positions[i] < positions[j] for i, j in pairs) / len(pairs)
                                                           if pairs else None)}
    return result


def _tokens(text):
    return len(TOKEN.findall(text)), len(NUMBER.findall(text))


def _worker(engine, data, output_path, timeout):
    import time

    started = time.monotonic()
    try:
        completed = subprocess.run([sys.executable, str(WORKER), engine, str(output_path or "-")],
                                   input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   timeout=timeout, check=False, env={"PATH": os.environ.get("PATH", "")})
    except subprocess.TimeoutExpired:
        if output_path:
            Path(output_path).unlink(missing_ok=True)
        return {"status": "timeout", "error_type": "TimeoutError",
                "elapsed_seconds": round(timeout, 6), "rss_max_bytes_approx": None}
    try:
        result = json.loads(completed.stdout) if completed.returncode == 0 else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        result = None
    if not isinstance(result, dict) or result.get("status") not in {"success", "failed"}:
        if output_path:
            Path(output_path).unlink(missing_ok=True)
        return {"status": "failed", "error_type": "WorkerError",
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "rss_max_bytes_approx": None}
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    if result["status"] == "failed" and output_path:
        Path(output_path).unlink(missing_ok=True)
    return result


def _versions():
    packages = {}
    for name in ("markitdown", "pdfplumber", "pdfminer.six"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return packages


def _private_dir(path):
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
    except OSError:
        raise BenchmarkInputError("output_permissions_unavailable") from None


def _write_private(path, text):
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
    except OSError:
        raise BenchmarkInputError("output_unavailable") from None


def run(manifest_path, pdf_root, output_root, timeout=30):
    cases = _manifest(manifest_path)
    try:
        root = Path(pdf_root).resolve(strict=True)
    except (OSError, RuntimeError):
        raise BenchmarkInputError("invalid_pdf_root") from None
    if not root.is_dir():
        raise BenchmarkInputError("invalid_pdf_root")
    try:
        output_base = Path(output_root).resolve()
    except (OSError, RuntimeError, ValueError):
        raise BenchmarkInputError("invalid_output_root") from None
    if output_base == root or output_base.is_relative_to(root):
        raise BenchmarkInputError("output_inside_pdf_root")
    resolved = [(case, _case_path(root, case["pdf"])) for case in cases]

    run_id = uuid.uuid4().hex
    run_dir = output_base / f"run-{run_id}"
    _private_dir(run_dir)
    _private_dir(run_dir / "raw")
    for directory in (run_dir / "raw" / "markitdown", run_dir / "raw" / "pdfplumber"):
        _private_dir(directory)

    records, errors, input_digests = [], [], {}
    for case, pdf_path in resolved:
        case_id, gold = case["case_id"], case["gold"]
        record = {"case_id": case_id, "page_count": None, "extractors": {}}
        try:
            data = _read_pdf(root, case["pdf"])
        except OSError as error:
            error_type = type(error).__name__
            error_type = error_type if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", error_type) else "OSError"
            record["input_status"] = "input_error"
            errors.append({"case_id": case_id, "extractor": "input", "status": "input_error",
                           "error_type": error_type})
            records.append(record)
            continue
        input_digests[case_id] = hashlib.sha256(data).hexdigest()
        inspection = _worker("inspect", data, None, timeout)
        if inspection.get("status") == "success":
            record["page_count"] = inspection.get("page_count")
        else:
            record["page_count_error_type"] = inspection.get("error_type", "UnknownError")
            errors.append({"case_id": case_id, "extractor": "page_count", "status": inspection["status"],
                           "error_type": inspection.get("error_type", "UnknownError")})

        texts = {}
        for engine, suffix in (("markitdown", ".md"), ("pdfplumber", ".txt")):
            raw_path = run_dir / "raw" / engine / f"{case_id}{suffix}"
            result = _worker(engine, data, raw_path, timeout)
            text = None
            if result["status"] == "success":
                text = raw_path.read_text(encoding="utf-8")
                result["status"] = "empty" if not text.strip() else "success"
                result["output_characters"] = len(text)
                result["token_count"], result["number_count"] = _tokens(text)
                if gold is not None:
                    result["gold"] = _gold_metrics(text, gold)
                texts[engine] = text
            record["extractors"][engine] = result
            if result["status"] not in {"success", "empty"}:
                errors.append({"case_id": case_id, "extractor": engine, "status": result["status"],
                               "error_type": result.get("error_type", "UnknownError")})
        if len(texts) == 2:
            left_tokens, left_numbers = _tokens(texts["markitdown"])
            right_tokens, right_numbers = _tokens(texts["pdfplumber"])
            record["descriptive_difference"] = {
                "character_delta_markitdown_minus_pdfplumber": len(texts["markitdown"]) - len(texts["pdfplumber"]),
                "token_delta_markitdown_minus_pdfplumber": left_tokens - right_tokens,
                "number_delta_markitdown_minus_pdfplumber": left_numbers - right_numbers,
            }
        records.append(record)

    statuses = {engine: {} for engine in ("markitdown", "pdfplumber")}
    for record in records:
        for engine in statuses:
            status = record.get("extractors", {}).get(engine, {}).get("status", record.get("input_status", "input_error"))
            statuses[engine][status] = statuses[engine].get(status, 0) + 1
    elapsed_by_engine = {engine: [record["extractors"][engine]["elapsed_seconds"] for record in records
                                  if engine in record.get("extractors", {}) and
                                  record["extractors"][engine].get("status") in {"success", "empty"}]
                         for engine in statuses}
    run_metadata = {"schema_version": 1, "run_id": run_id,
                    "benchmark_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(),
                    "platform": {"system": platform.system(), "release": platform.release(),
                                 "machine": platform.machine()},
                    "packages": _versions(), "configuration": {
                        "markitdown_enable_plugins": False,
                        "pdfplumber_text_method": r"Page.extract_text() defaults; separator \n\f\n",
                        "timeout_seconds_per_document_extractor": timeout,
                        "ocr": False, "network_services": False,
                        "raw_outputs_private": True}}
    run_metadata["input_sha256_by_case_id"] = input_digests
    run_metadata["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_metadata["worker_sha256"] = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    metrics = {"schema_version": 1, "run_id": run_id, "case_count": len(records),
               "status_counts": statuses, "extractor_median_elapsed_seconds": {
                   engine: statistics.median(values) if values else None for engine, values in elapsed_by_engine.items()},
               "cases": records}
    for path, value in ((run_dir / "run.json", run_metadata), (run_dir / "metrics.json", metrics),
                        (run_dir / "errors.json", {"errors": errors})):
        _write_private(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return run_dir, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("pdf_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".local-extraction-benchmark"))
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be greater than 0 and at most 600 seconds")
    try:
        _, metrics = run(args.manifest, args.pdf_root, args.output, args.timeout)
    except BenchmarkInputError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"Benchmark run {metrics['run_id']}: {metrics['case_count']} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
