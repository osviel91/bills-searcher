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
