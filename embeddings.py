import math
import os
import json
from pathlib import Path

import httpx

BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://host.docker.internal:8800/v1").rstrip("/")
MODEL = os.getenv("EMBEDDING_MODEL", "embedding")
API_KEY = os.getenv("EMBEDDING_API_KEY", "")
TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120"))
DIMENSIONS = 1024
QUERY_INSTRUCTION = "Instruct: Retrieve invoices relevant to the user's service or purchase query\nQuery:"
SETTINGS_FILE = Path(os.getenv("DATA_DIR", "/data")) / "embedding-settings.json"


def settings() -> dict:
    try:
        return {**{"base_url": BASE_URL, "model": MODEL, "api_key": API_KEY}, **json.loads(SETTINGS_FILE.read_text())}
    except (OSError, ValueError):
        return {"base_url": BASE_URL, "model": MODEL, "api_key": API_KEY}


def update_settings(base_url: str, model: str, api_key: str) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({"base_url": base_url.rstrip("/"), "model": model, "api_key": api_key}))


def available_models() -> list[str]:
    config = settings()
    headers = {"Authorization": f"Bearer {config['api_key']}"} if config["api_key"] else {}
    response = httpx.get(f"{config['base_url']}/models", headers=headers, timeout=10)
    response.raise_for_status()
    return sorted({item["id"] for item in response.json().get("data", []) if item.get("id")})


def embed(text: str, *, query: bool = False) -> list[float]:
    config = settings()
    headers = {"Authorization": f"Bearer {config['api_key']}"} if config["api_key"] else {}
    response = httpx.post(
        f"{config['base_url']}/embeddings",
        headers=headers,
        json={
            "model": config["model"],
            "input": f"{QUERY_INSTRUCTION} {text}" if query else text,
            "max_length": 32768,
            "truncation": False,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    vector = response.json()["data"][0]["embedding"]
    if len(vector) != DIMENSIONS or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in vector
    ):
        raise ValueError(f"El proveedor no devolvió un vector válido de {DIMENSIONS} dimensiones")
    return vector


def health() -> tuple[bool, bool]:
    config = settings()
    headers = {"Authorization": f"Bearer {config['api_key']}"} if config["api_key"] else {}
    try:
        response = httpx.get(f"{config['base_url']}/models", headers=headers, timeout=5)
        response.raise_for_status()
        models = {item.get("id") for item in response.json().get("data", [])}
        return True, config["model"] in models
    except (httpx.HTTPError, ValueError, TypeError):
        return False, False
