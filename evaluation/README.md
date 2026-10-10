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
  -v "$TEMPLATE_GOLD:/tmp/template-gold.json:ro" app \
  python -m scripts.evaluate_templates /tmp/template-gold.json /samples \
  --json /tmp/template-report.json --markdown /tmp/template-report.md
```

Keep the manifest outside Git and reports/shared artifacts; the recommended
directory is mode `0700` and the manifest mode `0600`. The manifest uses
`version: 1`; each case contains an opaque ID, relative PDF path, visually
confirmed class, line counts, field-presence labels, visual evidence boxes, and
optional business questions:

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
    "fields": {
      "invoice_number": {
        "present": true,
        "evidence": {
          "page": 1,
          "normalized_bbox": [0.05, 0.29, 0.20, 0.33],
          "region": "header",
          "granularity": "containing form region"
        }
      },
      "tax_amount": {"present": false, "evidence": null}
    },
    "business_questions": []
  }]
}
```

Required fields are `case_id`, `pdf`, `template` (`T01`, `T02`, or `unknown`),
`reviewed_visually`, and `commercial_line_count`. `fields` records
presence/absence independently of extractor predictions; present fields have
page and normalized `[x0, y0, x1, y1]` evidence. Reports contain opaque IDs,
classes, line counts, detector signals and abstention reasons only, and state
that metrics are sample results. To include the metrics in the Phase 2C
extraction report, pass `--templates /tmp/template-gold.json
--template-pdf-root /samples` to `scripts.evaluate_extraction` and mount the
private inputs read-only as above.
