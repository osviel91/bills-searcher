#!/usr/bin/env python3
"""Run the offline template-classification evaluation."""

import argparse
import json
from pathlib import Path

from .template_evaluation import evaluate_templates, markdown_template_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Private JSON gold manifest; keep it outside Git")
    parser.add_argument("pdf_root", type=Path, help="Directory containing the locally authorized PDFs")
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("template-report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("template-report.md"))
    args = parser.parse_args()
    report = evaluate_templates(args.manifest, args.pdf_root)
    args.json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    args.markdown.write_text(markdown_template_report(report))
    print(f"Wrote {args.json_path} and {args.markdown}")


if __name__ == "__main__":
    main()
