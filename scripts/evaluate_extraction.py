#!/usr/bin/env python3
"""Run the deterministic structured extraction evaluation dataset."""

import argparse
import json
from pathlib import Path

from .extraction_evaluation import evaluate, markdown_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("extraction-report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("extraction-report.md"))
    parser.add_argument("--templates", type=Path, help="Private template gold manifest")
    parser.add_argument("--template-pdf-root", type=Path, help="Directory containing template evaluation PDFs")
    args = parser.parse_args()
    if (args.templates is None) != (args.template_pdf_root is None):
        parser.error("--templates and --template-pdf-root must be supplied together")
    report = evaluate(args.dataset, args.templates, args.template_pdf_root)
    args.json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown.write_text(markdown_report(report))
    print(f"Wrote {args.json_path} and {args.markdown}")


if __name__ == "__main__":
    main()
