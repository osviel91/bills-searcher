#!/usr/bin/env python3
"""Live smoke test: verify queries return ranked invoices from the running app."""
import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8080"
QUERIES = (
    "facturas de mantenimiento",
    "reparación de equipos de oficina",
    "compra de material informático",
    "servicios de alojamiento web",
)


def main():
    for query in QUERIES:
        body = json.dumps({"query": query, "limit": 5}).encode()
        request = urllib.request.Request(f"{BASE_URL}/api/search", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            results = json.load(response)["results"]
        if not results:
            print(json.dumps({"query": query, "error": "no indexed results"}, ensure_ascii=False))
            return 1
        print(json.dumps({"query": query, "top5": [
            {"filename": item["filename"], "score": round(item["score"], 3)}
            for item in results
        ]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
