#!/usr/bin/env python3
"""Compute Recall@5/10 against human-labelled invoice filenames."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", help="JSON case file, or '-' to read cases from stdin")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    raw_cases = sys.stdin.read() if args.cases == "-" else Path(args.cases).read_text(encoding="utf-8")
    cases = json.loads(raw_cases)
    if not cases or any(not case.get("query") or not case.get("relevant") for case in cases):
        parser.error("cada caso debe tener query y al menos un filename relevante")

    scores = []
    top1_failures = 0
    for case in cases:
        payload = json.dumps({"query": case["query"], "limit": 10}).encode()
        request = urllib.request.Request(f"{args.base_url.rstrip('/')}/api/search", data=payload,
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                results = json.load(response)["results"]
        except (OSError, urllib.error.URLError, ValueError) as error:
            print(json.dumps({"query": case["query"], "error": str(error)}, ensure_ascii=False))
            return 1
        relevant = set(case["relevant"])
        filenames = [result["filename"] for result in results]
        recall5 = len(relevant.intersection(filenames[:5])) / len(relevant)
        recall10 = len(relevant.intersection(filenames[:10])) / len(relevant)
        top1_match = case.get("expected_top1") is None or bool(filenames and filenames[0] == case["expected_top1"])
        top1_failures += not top1_match
        scores.append((recall5, recall10))
        print(json.dumps({"query": case["query"], "top10": filenames,
                          "recall_at_5": recall5, "recall_at_10": recall10,
                          "expected_top1": case.get("expected_top1"), "top1_match": top1_match,
                          "latency_ms": round((time.perf_counter() - started) * 1000, 1)},
                         ensure_ascii=False))

    count = len(scores)
    print(json.dumps({"cases": count,
                      "mean_recall_at_5": sum(score[0] for score in scores) / count,
                      "mean_recall_at_10": sum(score[1] for score in scores) / count,
                      "top1_failures": top1_failures}))
    return 1 if top1_failures else 0


if __name__ == "__main__":
    sys.exit(main())
