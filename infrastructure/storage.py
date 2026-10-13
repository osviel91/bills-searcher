import os

from elasticsearch import Elasticsearch

INDEX = os.getenv("ELASTICSEARCH_INDEX", "invoices-poc-v1")
PASSAGES_INDEX = f"{INDEX}-passages"
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
ES_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")

INVOICE_FIELDS = {
    "invoice_number": {"type": "keyword"},
    "invoice_date": {"type": "date"},
    "customer_code": {"type": "keyword"},
    "customer_tax_id": {"type": "keyword"},
    "delivery_note_ids": {"type": "keyword"},
    "concepts": {"type": "text"},
    "currency": {"type": "keyword"},
    "total_payable": {"type": "scaled_float", "scaling_factor": 100},
    "invoice_total": {"type": "scaled_float", "scaling_factor": 100},
    "tax_base": {"type": "scaled_float", "scaling_factor": 100},
    "vat": {"type": "scaled_float", "scaling_factor": 100},
    "confidence": {"type": "object", "dynamic": True},
    "evidence": {"type": "object", "dynamic": True},
}


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
                "generation": {"type": "keyword"},
                "sha256": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "model": {"type": "keyword"},
                "content": {"type": "text"},
                **INVOICE_FIELDS,
                "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIMENSIONS, "index": True, "similarity": "cosine"},
            }})
        except Exception:
            if not es.indices.exists(index=INDEX):
                raise
    else:
        mapping = es.indices.get_mapping(index=INDEX)[INDEX]["mappings"]["properties"]
        actual_dimensions = mapping.get("embedding", {}).get("dims")
        if actual_dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"El índice {INDEX} tiene {actual_dimensions} dimensiones; configura uno nuevo para {EMBEDDING_DIMENSIONS}")
        es.indices.put_mapping(index=INDEX, properties={"content": {"type": "text"},
                                                       "generation": {"type": "keyword"}, **INVOICE_FIELDS})

    passage_properties = {
        "invoice_id": {"type": "keyword"}, "filename": {"type": "keyword"},
        "generation": {"type": "keyword"}, "active": {"type": "boolean"},
        "chunk_index": {"type": "integer"}, "start_offset": {"type": "integer"},
        "end_offset": {"type": "integer"}, "content": {"type": "text"},
        "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIMENSIONS,
                      "index": True, "similarity": "cosine"},
    }
    if not es.indices.exists(index=PASSAGES_INDEX):
        try:
            es.indices.create(index=PASSAGES_INDEX, mappings={"properties": passage_properties})
        except Exception:
            if not es.indices.exists(index=PASSAGES_INDEX):
                raise
    else:
        mapping = es.indices.get_mapping(index=PASSAGES_INDEX)[PASSAGES_INDEX]["mappings"]["properties"]
        actual_dimensions = mapping.get("embedding", {}).get("dims")
        if actual_dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"El índice {PASSAGES_INDEX} tiene {actual_dimensions} dimensiones; configura uno nuevo para {EMBEDDING_DIMENSIONS}")
        es.indices.put_mapping(index=PASSAGES_INDEX, properties=passage_properties)
