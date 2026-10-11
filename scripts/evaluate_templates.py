#!/usr/bin/env python3
"""Run the offline template-classification evaluation."""

import argparse
import json
from pathlib import Path

from .template_evaluation import (evaluate_template_lines, evaluate_templates,
                                  markdown_line_report, markdown_template_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Private JSON gold manifest; keep it outside Git")
    parser.add_argument("pdf_root", type=Path, help="Directory containing the locally authorized PDFs")
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("template-report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("template-report.md"))
    parser.add_argument("--lines", action="store_true",
                        help="Evaluate private line gold using offline AS/400 and generic extractors")
    parser.add_argument("--inventory", type=Path,
                        help="Private neutral geometry inventory (required with --lines)")
    parser.add_argument("--template-gold", type=Path,
                        help="Private header/template gold (required with --lines)")
    args = parser.parse_args()
    if args.lines:
        if args.inventory is None or args.template_gold is None:
            parser.error("--lines requires --inventory and --template-gold")
        report = evaluate_template_lines(args.manifest, args.pdf_root, args.inventory, args.template_gold)
        markdown = markdown_line_report(report)
    else:
        report = evaluate_templates(args.manifest, args.pdf_root)
        markdown = markdown_template_report(report)
    args.json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    args.markdown.write_text(markdown)
    print(f"Wrote {args.json_path} and {args.markdown}")


if __name__ == "__main__":
    main()
