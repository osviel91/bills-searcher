import base64
import math
import os
import json
from pathlib import Path

import httpx

BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://host.docker.internal:8800/v1").rstrip("/")
MODEL = os.getenv("EMBEDDING_MODEL", "embedding")
API_KEY = os.getenv("EMBEDDING_API_KEY", "")
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai-compatible")
DOCUMENT_MODE = os.getenv("EMBEDDING_DOCUMENT_MODE", "text")
TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "300"))
DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
QUERY_INSTRUCTION = "Instruct: Retrieve invoices relevant to the user's service or purchase query\nQuery:"
SETTINGS_FILE = Path(os.getenv("DATA_DIR", "/data")) / "embedding-settings.json"


def settings() -> dict:
    try:
        return {**{"base_url": BASE_URL, "model": MODEL, "api_key": API_KEY,
                    "provider": PROVIDER, "document_mode": DOCUMENT_MODE,
                    "dimensions": DIMENSIONS}, **json.loads(SETTINGS_FILE.read_text())}
    except (OSError, ValueError):
        return {"base_url": BASE_URL, "model": MODEL, "api_key": API_KEY,
                "provider": PROVIDER, "document_mode": DOCUMENT_MODE,
                "dimensions": DIMENSIONS}


def update_settings(base_url: str, model: str, api_key: str, provider: str = PROVIDER,
                    document_mode: str = "text") -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({"base_url": base_url.rstrip("/"), "model": model,
                                        "api_key": api_key, "provider": provider,
                                        "document_mode": document_mode}))


def available_models() -> list[str]:
    config = settings()
    headers = {"Accept": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    response = httpx.get(f"{config['base_url']}/models", headers=headers, timeout=10)
    response.raise_for_status()
    return sorted({item["id"] for item in response.json().get("data", []) if item.get("id")})


def embed(text: str, *, query: bool = False) -> list[float]:
    config = settings()
    headers = {"Accept": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    if config["provider"] == "jina-local":
        payload = {"model": config["model"], "input": text,
                   "input_type": "query" if query else "document", "dimensions": DIMENSIONS}
    else:
        payload = {
            "model": config["model"],
            "input": f"{QUERY_INSTRUCTION} {text}" if query else text,
            "max_length": 32768,
            "truncation": False,
        }
    response = httpx.post(
        f"{config['base_url']}/embeddings",
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    vector = response.json()["data"][0]["embedding"]
    if len(vector) != DIMENSIONS or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in vector
    ):
        raise ValueError(f"El proveedor no devolvió un vector válido de {DIMENSIONS} dimensiones")
    return vector


def embed_file(data: bytes, content_type: str, document_id: str) -> list[float]:
    config = settings()
    if config["provider"] != "jina-local" or config["document_mode"] != "direct":
        raise ValueError("La entrada directa de PDF/imagen requiere Jina local en modo multimodal")
    file_type = {"application/pdf": "pdf", "image/png": "image", "image/jpeg": "image",
                 "image/webp": "image"}.get(content_type)
    if not file_type:
        raise ValueError(f"Tipo de documento no admitido: {content_type}")
    item = {"type": file_type, "data": base64.b64encode(data).decode("ascii")}
    if file_type == "pdf":
        item["document_id"] = document_id
    headers = {"Accept": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    response = httpx.post(f"{config['base_url']}/embeddings", headers=headers,
                          json={"model": config["model"], "input": [item],
                                "input_type": "document", "dimensions": DIMENSIONS},
                          timeout=TIMEOUT)
    response.raise_for_status()
    vector = response.json()["data"][0]["embedding"]
    if len(vector) != DIMENSIONS or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in vector
    ):
        raise ValueError(f"El proveedor no devolvió un vector válido de {DIMENSIONS} dimensiones")
    return vector


def health() -> tuple[bool, bool]:
    config = settings()
    headers = {"Accept": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    try:
        response = httpx.get(f"{config['base_url']}/models", headers=headers, timeout=5)
        response.raise_for_status()
        models = {item.get("id") for item in response.json().get("data", [])}
        return True, config["model"] in models
    except (httpx.HTTPError, ValueError, TypeError):
        return False, False
