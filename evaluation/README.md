# Structured extraction dataset v1

Run from the repository root:

```sh
docker compose run --rm --no-deps -v "$PWD:/app" app \
  python -m scripts.evaluate_extraction evaluation/structured-extraction-v1.json
```

Optional report paths:

```sh
docker compose run --rm --no-deps -v "$PWD:/app" app \
  python -m scripts.evaluate_extraction evaluation/structured-extraction-v1.json \
  --json /tmp/extraction-report.json --markdown /tmp/extraction-report.md
```

The JSON root has `version: 1` and a `cases` array. Each case has a stable
`case_id`, a repository-relative PDF path, optional `template`, a complete
`expected_fields` map (use `null` for intentionally absent values), optional
`intentionally_absent`, `expected_lines`, and `evidence` keyed by field. An
evidence reference is `{ "page": 1, "quote": "exact source text" }`. Lines
contain a description and attribute map; evaluation matches normalized full
line content, not position. Money uses exact decimal equality.

Keep PDFs synthetic or anonymized. Reports intentionally contain case IDs,
field names, and error types only, never extracted values, quotes, filenames,
or source document text.

## Template classification

Template detectors are evaluation-only; they are not registered in the active
extractor selection. Run the classifier against a private gold manifest and a
locally authorized PDF directory:

Positive classification requires two independent signals: an exact ordered
column header plus either two family-specific fixed labels or a matching
normalized detail-region geometry. Only one-page portrait PDFs with aspect
ratio 0.68-0.73 are supported; ties, missing signals, and incompatible pages
abstain as `unknown`.

```sh
TEMPLATE_GOLD="$HOME/.local/share/bills-searcher/template-evaluation/gold-v1.json"
AUTHORIZED_PDFS="/path/to/authorized-pdfs"

docker compose run --rm --no-deps -v "$PWD:/app" \
  -v "$AUTHORIZED_PDFS:/samples:ro" \
  -v "$TEMPLATE_GOLD:/private/gold-v1.json:ro" app \
  python -m scripts.evaluate_templates /private/gold-v1.json /samples \
  --json /tmp/template-report.json --markdown /tmp/template-report.md
```

Keep the manifest outside Git and reports/shared artifacts; the recommended
directory is mode `0700` and the manifest mode `0600`. The manifest uses
`version: 1`; each case contains an opaque ID, relative PDF path, visually
confirmed class, line counts, reviewed field labels, visual evidence boxes, and
optional business questions. For Phase 2B.2-B, each field reference adds a
state and a visually transcribed normalized value:

```json
{
  "version": 1,
  "cases": [{
    "case_id": "doc-opaque-01",
    "pdf": "document.pdf",
    "template": "T01",
    "reviewed_visually": true,
    "commercial_line_count": 2,
    "visible_detail_rows": 3,
    "fields_reviewed_visually": true,
    "fields": {
      "issue_date": {
        "status": "confirmed",
        "expected": "YYYY-MM-DD",
        "evidence": {
          "page": 1,
          "value_quote": "verbatim source date",
          "value_bbox": [0.21, 0.29, 0.32, 0.31],
          "source_token_ids": ["p1-wN-cN"],
          "method": "header-label-same-column",
          "label_quote": "FECHA",
          "label_bbox": [0.24, 0.32, 0.28, 0.33]
        }
      },
      "total_due": {"status": "ambiguous", "expected": null, "evidence": null}
    },
    "business_questions": []
  }]
}
```

Required fields are `case_id`, `pdf`, `template` (`T01`, `T02`, or `unknown`),
`reviewed_visually`, `commercial_line_count`, `fields_reviewed_visually`, and
the per-field `status`, `expected`, and `evidence`. Valid field states are
`confirmed`, `absent`, `ambiguous`, `not_applicable`,
`unreadable`, `conflict`, `insufficient_evidence`, and
`unverified_semantics`. Only `confirmed` fields enter exactness/coverage
denominators; the latter state is reported as a documentary candidate, not a
financial prediction. The evaluator compares both `as400` and `generic` on the
same references, by template and field, and reports value precision/recall,
coverage, selective accuracy, evidence precision/recall, role errors, amount
association errors, false positives, false negatives, and abstentions. Reports
never contain reference values, quotes, filenames, or private paths. Results
are sample metrics, not a general accuracy estimate. The existing Phase 2C
runner can include these metrics by adding `--templates /private/gold-v1.json
--template-pdf-root /samples` and mounting both private inputs read-only.

## AS/400 sample result

Visually reviewed sample metrics from Phase 2B.2-B (not a corpus-level
accuracy estimate):

| Template | Extractor | Confirmed fields | Exact | Precision | Recall | Coverage | Evidence P/R | FP/FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T01 | `as400-t01-header-summary@1` | 27 | 27 | 1.000 | 1.000 | 1.000 | 1.000 / 1.000 | 0 / 0 |
| T01 | `generic-invoice@1` baseline | 27 | 0 | 0.000 | 0.000 | 0.111 | 0.000 / 0.000 | 3 / 27 |
| T02 | `as400-t02-header-summary@1` | 74 | 74 | 1.000 | 1.000 | 1.000 | 1.000 / 1.000 | 0 / 0 |
| T02 | `generic-invoice@1` baseline | 74 | 7 | 0.438 | 0.095 | 0.216 | 0.000 / 0.000 | 9 / 67 |

These metrics exclude ambiguous totals and `not_applicable` VAT-rate fields.
The legacy `TOTAL EUROS` amount remains an unassigned documentary candidate,
not an invoice or payable total. One visually distinct document is `unknown`
and receives no AS/400 extractor. Sample reports remain local and use opaque
document IDs without source values or citations.
