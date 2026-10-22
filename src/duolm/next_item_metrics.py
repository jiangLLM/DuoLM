"""User-macro next-item metrics with a fixed cohort and retrieval misses intact."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Sequence


def ranking_metrics(rows: Iterable[tuple[object, Sequence[float]]], *,
                    expected_user_ids: Sequence[str], unavailable_user_ids: Sequence[str] = (),
                    k: int = 10) -> dict:
    """Average queries within each user, then users; never condition on a hit.

    A retrieval miss contributes zero to every ranking metric. Score ties are
    resolved by item ID, so packing/batching cannot change tie order.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    expected = tuple(expected_user_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected users must be nonempty and unique")
    expected_set = set(expected)
    unavailable = set(unavailable_user_ids)
    if len(unavailable) != len(unavailable_user_ids) or not unavailable <= expected_set:
        raise ValueError("unavailable users must be unique members of the fixed cohort")
    per_user, seen = defaultdict(list), set()
    for query, scores in rows:
        if query.query_id in seen:
            raise ValueError("duplicate evaluation query ID")
        seen.add(query.query_id)
        if query.user_id not in expected_set:
            raise ValueError("evaluation contains a user outside the fixed cohort")
        if query.user_id in unavailable:
            raise ValueError("an unavailable user cannot also have an evaluation query")
        values = [float(value) for value in scores]
        if len(values) != len(query.candidate_ids) or not all(math.isfinite(v) for v in values):
            raise ValueError("scores must match every candidate and be finite")
        hit = query.positive_index is not None
        rank = None
        if hit:
            if isinstance(query.positive_index, bool) or not isinstance(query.positive_index, int) or not 0 <= query.positive_index < len(values):
                raise ValueError("invalid positive candidate index")
            order = sorted(range(len(values)), key=lambda i: (-values[i], query.candidate_ids[i]))
            rank = order.index(query.positive_index) + 1
        top_hit = rank is not None and rank <= k
        per_user[query.user_id].append({f"ndcg@{k}": 1 / math.log2(rank + 1) if top_hit else 0.0,
                                      f"hr@{k}": float(top_hit),
                                      f"mrr@{k}": 1 / rank if top_hit else 0.0,
                                      "retrieval_recall": float(hit)})
    for user in unavailable:
        per_user[user].append({f"ndcg@{k}": 0.0, f"hr@{k}": 0.0, f"mrr@{k}": 0.0,
                               "retrieval_recall": 0.0})
    if set(per_user) != expected_set:
        missing = len(expected_set - set(per_user))
        raise ValueError(f"fixed evaluation cohort is incomplete: {missing} users missing")
    keys = (f"ndcg@{k}", f"hr@{k}", f"mrr@{k}", "retrieval_recall")
    result = {key: sum(sum(row[key] for row in per_user[user]) / len(per_user[user])
                       for user in expected) / len(expected) for key in keys}
    result.update(users=len(expected), queries=len(seen), unavailable_users=len(unavailable),
                  aggregation="user_macro", k=k)
    return result
