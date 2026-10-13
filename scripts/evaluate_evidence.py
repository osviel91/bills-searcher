#!/usr/bin/env python3
"""Measure evidence retrieval against human-labelled invoice quotes."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domain.evidence_evaluation import recall_at_k, valid_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="JSON case file, or '-' to read cases from stdin")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token-env", default="API_TOKEN", help="environment variable containing the API bearer token")
    parser.add_argument("--limit", type=int, default=10, help="number of passages to retrieve (10..20)")
    parser.add_argument("--timeout", type=float, default=900, help="per-query HTTP timeout in seconds")
    args = parser.parse_args()
    raw_cases = sys.stdin.read() if args.cases == "-" else Path(args.cases).read_text(encoding="utf-8")
    cases = json.loads(raw_cases)
    if not valid_cases(cases):
        parser.error("cada caso debe tener query y relevant_evidence con invoice_id y quote")
    if not 10 <= args.limit <= 20:
        parser.error("--limit debe estar entre 10 y 20 para medir Recall@5 y Recall@10")
    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"falta la variable de entorno {args.token_env}")

    scores = []
    for case in cases:
        payload = json.dumps({"query": case["query"], "limit": args.limit}).encode()
        request = urllib.request.Request(
            f"{args.base_url.rstrip('/')}/api/search/evidence", data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                results = json.load(response)["evidence"]
        except (OSError, urllib.error.URLError, ValueError, KeyError) as error:
            print(json.dumps({"query": case["query"], "error": str(error)}, ensure_ascii=False))
            return 1

        relevant = case["relevant_evidence"]
        recall5 = recall_at_k(relevant, results, 5)
        recall10 = recall_at_k(relevant, results, 10)
        scores.append((recall5, recall10))
        print(json.dumps({
            "query": case["query"],
            "relevant_evidence": len(relevant),
            "recall_at_5": recall5,
            "recall_at_10": recall10,
            "top10": [{key: result.get(key) for key in (
                "invoice_id", "filename", "chunk_index", "start_offset", "end_offset", "score",
            )} for result in results[:10]],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }, ensure_ascii=False))

    count = len(scores)
    print(json.dumps({
        "cases": count,
        "mean_recall_at_5": sum(score[0] for score in scores) / count,
        "mean_recall_at_10": sum(score[1] for score in scores) / count,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
