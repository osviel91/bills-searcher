import os

from elasticsearch import Elasticsearch

INDEX = os.getenv("ELASTICSEARCH_INDEX", "invoices-poc-v1")
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
ES_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")


def client():
    auth = {"api_key": ES_API_KEY} if ES_API_KEY else (
        {"basic_auth": (ES_USERNAME, ES_PASSWORD)} if ES_USERNAME and ES_PASSWORD else {}
    )
    return Elasticsearch(ES_URL, request_timeout=10, **auth)


def initialize_index(es):
    if not es.indices.exists(index=INDEX):
        try:
            es.indices.create(index=INDEX, mappings={"properties": {
                "invoice_id": {"type": "keyword"},
                "sha256": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "model": {"type": "keyword"},
                "content": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"},
            }})
        except Exception:
            if not es.indices.exists(index=INDEX):
                raise
    else:
        es.indices.put_mapping(index=INDEX, properties={"content": {"type": "text"}})
