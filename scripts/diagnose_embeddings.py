#!/usr/bin/env python3
"""Check service/model discovery; deliberately sends no undocumented PDF request."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

base = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8800/v1").rstrip("/")
model = os.getenv("EMBEDDING_MODEL", "embedding")
key = os.getenv("EMBEDDING_API_KEY", "")
headers = {"Authorization": f"Bearer {key}"} if key else {}
started = time.monotonic()
try:
    req = urllib.request.Request(f"{base}/models", headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        data = json.load(response)
except (OSError, urllib.error.URLError) as error:
    print(json.dumps({"status": "blocked", "stage": "service", "error": str(error), "model": model, "elapsed_seconds": round(time.monotonic()-started, 3)}))
    sys.exit(2)

available = [entry.get("id") for entry in data.get("data", [])]
if model not in available:
    print(json.dumps({"status": "blocked", "stage": "model", "model": model,
                      "model_available": False, "available_model_ids": available,
                      "elapsed_seconds": round(time.monotonic()-started, 3)}))
    sys.exit(3)

body = json.dumps({"model": model, "input": "Query: factura de mantenimiento"}).encode()
headers["Content-Type"] = "application/json"
req = urllib.request.Request(f"{base}/embeddings", data=body, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120"))) as response:
        result = json.load(response)
    vector = result["data"][0]["embedding"]
    query_result = {"query_embedding": "ok", "query_dimensions": len(vector),
                    "numeric": all(isinstance(value, (int, float)) for value in vector)}
except urllib.error.HTTPError as error:
    detail = error.read().decode(errors="replace")
    query_result = {"query_embedding": "blocked", "http_status": error.code,
                    "detail": detail[:1000]}
except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
    query_result = {"query_embedding": "error", "error": str(error)}

ready = query_result.get("query_embedding") == "ok" and query_result.get("numeric")
print(json.dumps({"status": "ready_text_mode" if ready else "blocked",
                  "stage": "text_embedding" if ready else "query_embedding",
                  "model": model, "model_available": True, **query_result,
                  "pdf_processing": "MarkItDown text extraction in app; this probe sends no PDF",
                  "elapsed_seconds": round(time.monotonic()-started, 3)}))
sys.exit(0 if ready else 3)
