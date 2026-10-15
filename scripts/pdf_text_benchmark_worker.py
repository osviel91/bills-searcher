"""Private subprocess adapter for the local PDF text benchmark."""

import contextlib
import json
import os
import re
import resource
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path


def _rss_bytes():
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (AttributeError, OSError, ValueError):
        return None


def _run(engine, data, output_path):
    started = time.monotonic()
    if engine == "inspect" or engine == "pdfplumber":
        import pdfplumber

        with pdfplumber.open(BytesIO(data)) as pdf:
            pages = len(pdf.pages)
            if engine == "pdfplumber":
                text = "\n\f\n".join(page.extract_text() or "" for page in pdf.pages)
            else:
                text = None
    elif engine == "markitdown":
        from markitdown import MarkItDown

        with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf:
            pdf.write(data)
            pdf.flush()
            text = MarkItDown(enable_plugins=False).convert_local(pdf.name).markdown.strip()
        pages = None
    else:
        raise ValueError("InvalidEngine")

    if output_path is not None:
        path = Path(output_path)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
    return {"status": "success", "page_count": pages, "elapsed_seconds": time.monotonic() - started,
            "rss_max_bytes_approx": _rss_bytes(), "output_characters": len(text) if text is not None else None}


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else ""
    output_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    result = None
    try:
        data = sys.stdin.buffer.read()
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                result = _run(engine, data, output_path)
    except Exception as error:
        error_type = type(error).__name__
        result = {"status": "failed", "error_type": error_type if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", error_type) else "Exception"}
    sys.__stdout__.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
