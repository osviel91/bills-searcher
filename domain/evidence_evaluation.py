def normalize(text):
    return " ".join(text.casefold().split())


def recall_at_k(relevant, results, k):
    retrieved = results[:k]
    hits = sum(any(
        result.get("invoice_id") == expected["invoice_id"]
        and normalize(expected["quote"]) in normalize(result.get("content", ""))
        for result in retrieved
    ) for expected in relevant)
    return hits / len(relevant)


def invoice_recall_at_k(relevant_invoice_ids, results, k):
    retrieved_ids = {result.get("invoice_id") for result in results[:k]}
    return len(set(relevant_invoice_ids).intersection(retrieved_ids)) / len(set(relevant_invoice_ids))


def scoped_recall_at_k(relevant, results_by_invoice, k):
    hits = sum(any(
        normalize(expected["quote"]) in normalize(result.get("content", ""))
        for result in results_by_invoice.get(expected["invoice_id"], [])[:k]
    ) for expected in relevant)
    return hits / len(relevant)


def valid_cases(cases):
    return isinstance(cases, list) and bool(cases) and all(
        isinstance(case, dict) and isinstance(case.get("query"), str) and case["query"].strip()
        and isinstance(case.get("relevant_evidence"), list) and case["relevant_evidence"]
        and all(isinstance(item, dict) and isinstance(item.get("invoice_id"), str)
                and item["invoice_id"].strip() and isinstance(item.get("quote"), str)
                and item["quote"].strip() for item in case["relevant_evidence"])
        for case in cases
    )
