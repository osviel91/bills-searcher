import os
from pathlib import Path

from infrastructure.embeddings import embed
from infrastructure.jobstore import visible_generation_query
from infrastructure.storage import INDEX, PASSAGES_INDEX, client as es

DB = Path(os.getenv("DATA_DIR", "/data")) / "jobs.sqlite3"
INVOICE_SEARCH_FIELDS = [
    "invoice_id", "filename", "invoice_number", "invoice_date", "customer_code",
    "customer_tax_id", "delivery_note_ids", "concepts", "currency", "total_payable",
    "invoice_total", "tax_base", "vat", "confidence", "evidence",
]


class ServiceUnavailable(Exception):
    """An application service cannot complete its request due to a dependency failure."""


def reciprocal_rank_fusion(*rankings, limit):
    fused = {}
    for hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            item = fused.setdefault(hit["_id"], {"_id": hit["_id"], "_source": hit["_source"], "_score": 0.0})
            item["_score"] += 1 / (60 + rank)
    return sorted(fused.values(), key=lambda item: item["_score"], reverse=True)[:limit]


def run_search(query: str, limit: int):
    try:
        es_client = es()
        visible = visible_generation_query(path=DB)
        window = max(50, limit * 10)
        lexical = es_client.search(index=INDEX, size=window, query={"bool": {"filter": [
            visible,
        ], "should": [
            {"match": {"content": {"query": query, "operator": "or"}}},
            {"match_phrase": {"content": {"query": query, "slop": 3, "boost": 4}}},
        ], "minimum_should_match": 1}}, source=INVOICE_SEARCH_FIELDS)
    except Exception as error:
        raise ServiceUnavailable(f"No se pudo completar la búsqueda: {type(error).__name__}") from error
    try:
        vector = embed(query, query=True)
        semantic = es_client.search(index=INDEX, size=window, knn={
            "field": "embedding", "query_vector": vector,
            "k": window, "num_candidates": max(window, min(1000, limit * 10)),
            "filter": visible,
        }, source=INVOICE_SEARCH_FIELDS)
        rankings = (lexical["hits"]["hits"], semantic["hits"]["hits"])
    except Exception:
        rankings = (lexical["hits"]["hits"],)
    hits = reciprocal_rank_fusion(*rankings, limit=limit)
    return {"results": [
        {**hit["_source"], "score": hit["_score"], "file_url": f"/api/documents/{hit['_source']['invoice_id']}/file"}
        for hit in hits
    ]}


def run_invoice_filter(*, customer_code=None, customer_tax_id=None, invoice_number=None,
                       delivery_note_id=None, concept=None, currency=None, invoice_date_from=None,
                       invoice_date_to=None, total_payable_min=None, total_payable_max=None,
                       limit=50, offset=0):
    filters = []
    for field, value in (("customer_code", customer_code), ("customer_tax_id", customer_tax_id),
                         ("invoice_number", invoice_number), ("delivery_note_ids", delivery_note_id),
                         ("currency", currency)):
        if value:
            filters.append({"term": {field: value}})
    if invoice_date_from or invoice_date_to:
        date_range = {}
        if invoice_date_from:
            date_range["gte"] = invoice_date_from
        if invoice_date_to:
            date_range["lte"] = invoice_date_to
        filters.append({"range": {"invoice_date": date_range}})
    if total_payable_min is not None or total_payable_max is not None:
        amount_range = {}
        if total_payable_min is not None:
            amount_range["gte"] = total_payable_min
        if total_payable_max is not None:
            amount_range["lte"] = total_payable_max
        filters.append({"range": {"total_payable": amount_range}})
    if concept:
        filters.append({"match": {"concepts": {"query": concept, "operator": "and"}}})
    try:
        response = es().search(
            index=INDEX, from_=offset, size=limit,
            query={"bool": {"filter": [*filters, visible_generation_query(path=DB)]}},
            source=INVOICE_SEARCH_FIELDS,
        )
    except Exception as error:
        raise ServiceUnavailable(f"No se pudo filtrar facturas: {type(error).__name__}") from error
    return {"results": [
                {**hit["_source"], "file_url": f"/api/documents/{hit['_source']['invoice_id']}/file"}
                for hit in response["hits"]["hits"]
            ],
            "total": response["hits"]["total"]["value"], "limit": limit, "offset": offset}


def run_invoice_evidence_search(query: str, limit: int = 5, invoice_id: str | None = None):
    try:
        es_client = es()
        visible = visible_generation_query(passages=True, path=DB)
        filters = [visible]
        if invoice_id:
            filters.append({"term": {"invoice_id": invoice_id}})
        passage_filter = {"bool": {"filter": filters}}
        window = max(20, limit * 5)
        lexical = es_client.search(
            index=PASSAGES_INDEX, size=window,
            query={"bool": {"filter": filters, "should": [
                {"match": {"content": {"query": query, "operator": "or"}}},
                {"match_phrase": {"content": {"query": query, "slop": 3, "boost": 4}}},
            ], "minimum_should_match": 1}},
            source=["invoice_id", "filename", "chunk_index", "start_offset", "end_offset", "content"],
        )
    except Exception as error:
        raise ServiceUnavailable(f"No se pudo recuperar evidencia: {type(error).__name__}") from error
    try:
        vector = embed(query, query=True)
        semantic = es_client.search(
            index=PASSAGES_INDEX, size=window,
            knn={"field": "embedding", "query_vector": vector, "k": window,
                 "num_candidates": max(window, min(1000, limit * 10)), "filter": passage_filter},
            source=["invoice_id", "filename", "chunk_index", "start_offset", "end_offset", "content"],
        )
        rankings = (lexical["hits"]["hits"], semantic["hits"]["hits"])
    except Exception:
        rankings = (lexical["hits"]["hits"],)
    hits = reciprocal_rank_fusion(*rankings, limit=limit)
    return {"evidence": [
        {**hit["_source"], "passage_id": hit["_id"], "score": hit["_score"],
         "file_url": f"/api/documents/{hit['_source']['invoice_id']}/file"}
        for hit in hits
    ]}
