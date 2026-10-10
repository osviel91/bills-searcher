"""Offline evaluation of registered invoice-template detectors."""

import json
from pathlib import Path

from application.template_detection import detect_template
from infrastructure.pdf_extraction import extract_pdf


CLASSES = ("T01", "T02", "unknown")


def evaluate_templates(manifest_path, pdf_root):
    manifest_path = Path(manifest_path).resolve()
    root = Path(pdf_root).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported template evaluation dataset version")

    cases = []
    for case in manifest.get("cases", []):
        case_id, expected, relative_pdf = case.get("case_id"), case.get("template"), case.get("pdf")
        line_count = case.get("commercial_line_count")
        if (not isinstance(case_id, str) or expected not in CLASSES or not isinstance(relative_pdf, str)
                or not case.get("reviewed_visually") or not isinstance(line_count, int) or line_count < 0):
            raise ValueError("Each case requires visual review, an opaque case_id, template, line count, and PDF path")
        pdf_path = (root / relative_pdf).resolve()
        if root not in pdf_path.parents or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path must stay inside the sample directory in case {case_id}")
        result = detect_template(extract_pdf(pdf_path.read_bytes()))
        cases.append({"case_id": case_id, "expected": expected, "predicted": result.template,
                      "commercial_line_count": line_count,
                      "visible_detail_rows": case.get("visible_detail_rows"),
                      "business_validation_required": bool(case.get("business_questions")),
                      "detector_id": result.detector_id, "detector_version": result.detector_version,
                      "signals": result.signals, "reason": result.reason})

    matrix = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    for case in cases:
        matrix[case["expected"]][case["predicted"]] += 1
    accuracy = {}
    for template in CLASSES:
        support = sum(matrix[template].values())
        accuracy[template] = {"correct": matrix[template][template], "support": support,
                              "accuracy": matrix[template][template] / support if support else None}
    per_template = {template: {
        "false_positives": sum(matrix[actual][template] for actual in CLASSES if actual != template),
        "false_negatives": sum(matrix[template][predicted] for predicted in CLASSES if predicted != template),
    } for template in ("T01", "T02")}
    return {"version": 1,
            "classification_rule": "At least 2 signals: exact ordered header plus >=2 fixed labels or matching geometry; one-page portrait ratio 0.68-0.73; multiple matches abstain.",
            "cases": cases,
            "metrics": {"confusion_matrix": matrix, "accuracy_by_template": accuracy,
                        "false_positives_and_negatives": per_template,
                        "abstentions": sum(case["predicted"] == "unknown" for case in cases),
                        "abstentions_on_known_templates": sum(
                            case["expected"] != "unknown" and case["predicted"] == "unknown" for case in cases),
                        "ambiguous_cases": [case["case_id"] for case in cases
                                            if case["reason"] == "ambiguous-template-match"]}}


def markdown_template_report(report):
    metrics = report["metrics"]
    labels = CLASSES
    lines = ["# Template Detection Evaluation", "",
             "Rule: " + report["classification_rule"], "", "## Confusion matrix", "",
             "| Reference \\ Prediction | " + " | ".join(labels) + " |",
             "|---|---:|---:|---:|"]
    lines.extend("| " + actual + " | " + " | ".join(str(metrics["confusion_matrix"][actual][p]) for p in labels) + " |"
                 for actual in labels)
    lines.extend(["", "## Per-template metrics", "", "| Template | Accuracy | False positives | False negatives |",
                  "|---|---:|---:|---:|"])
    for template in ("T01", "T02", "unknown"):
        score = metrics["accuracy_by_template"][template]
        errors = metrics["false_positives_and_negatives"].get(template, {"false_positives": 0, "false_negatives": 0})
        accuracy = "n/a" if score["accuracy"] is None else f"{score['accuracy']:.3f} ({score['correct']}/{score['support']})"
        lines.append(f"| {template} | {accuracy} | {errors['false_positives']} | {errors['false_negatives']} |")
    lines.extend(["", f"Abstentions: {metrics['abstentions']}",
                  f"Known-template abstentions: {metrics['abstentions_on_known_templates']}",
                  "Ambiguous cases: " + (", ".join(f"`{item}`" for item in metrics["ambiguous_cases"]) or "none"),
                  "", "## Document diagnostics", "", "| Opaque ID | Reference | Prediction | Commercial lines | Business review | Detector | Reason | Signals |",
                  "|---|---|---:|---:|---|---|---|---|"])
    for case in report["cases"]:
        signals = "; ".join(
            detector + "[" + ",".join(f"{name}={','.join(values) or '-'}" for name, values in groups.items()) + "]"
            for detector, groups in case["signals"].items())
        lines.append(f"| `{case['case_id']}` | {case['expected']} | {case['predicted']} | "
                     f"{case['commercial_line_count']} | {'required' if case['business_validation_required'] else 'no'} | "
                     f"{case['detector_id']}@{case['detector_version']} | {case['reason'] or 'identified'} | {signals} |")
    lines.extend(["", "Sample results only; not an estimate of general system accuracy.", ""])
    return "\n".join(lines)
