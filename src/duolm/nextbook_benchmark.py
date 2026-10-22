"""Label-free dev workloads and numerical gates for next-book serving.

Capacity accounting is the unchanged open-loop rating benchmark contract. A
scalar ranker has its own numerical contract; no rating probabilities are used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

PROTOCOL = "mixlm-nextbook-capacity-v1"
SCORE_ATOL = 0.02
PROBABILITY_ATOL = 0.001


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def softmax(scores):
    offset = max(scores)
    values = [math.exp(x - offset) for x in scores]
    total = sum(values)
    return [x / total for x in values]


def numerical_check(actual, reference):
    if (not isinstance(actual, list) or not isinstance(reference, list)
            or len(actual) != len(reference) or not actual
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in actual + reference)):
        raise ValueError("Wrong or nonfinite scalar score group")
    maximum = max(abs(a - b) for a, b in zip(actual, reference))
    probability = max(abs(a - b) for a, b in zip(softmax(actual), softmax(reference)))
    # Pairs separated by more than twice the absolute error budget may not flip.
    # Near-tie changes are reported rather than misclassified as attention bugs.
    flips = sum((reference[i] - reference[j]) * (actual[i] - actual[j]) <= 0
                for i in range(len(actual)) for j in range(i)
                if abs(reference[i] - reference[j]) > 2 * SCORE_ATOL)
    rank_actual = sorted(range(len(actual)), key=lambda i: (-actual[i], i))
    rank_reference = sorted(range(len(reference)), key=lambda i: (-reference[i], i))
    return {"accepted": maximum <= SCORE_ATOL and probability <= PROBABILITY_ATOL and flips == 0,
            "max_score_difference": maximum, "max_group_probability_difference": probability,
            "separated_pair_flips": flips, "top1_equal": rank_actual[0] == rank_reference[0],
            "top10_overlap": len(set(rank_actual[:10]) & set(rank_reference[:10])),
            "score_atol": SCORE_ATOL, "group_probability_atol": PROBABILITY_ATOL}


def build_workload(dev_file, *, users=100, candidate_counts=(10, 50, 100), seed=17):
    if type(users) is not int or users < 1 or not candidate_counts or any(
            type(n) is not int or not 1 <= n <= 100 for n in candidate_counts):
        raise ValueError("Invalid workload inventory")
    path = Path(dev_file)
    if path.name != "dev.jsonl":
        raise ValueError("Only the frozen dev split may supply benchmark inputs")
    raw = path.read_bytes()
    choices = {}
    for line in raw.splitlines():
        row = json.loads(line)
        # Query selection never inspects target, positive index or model scores.
        if len(row["history"]) != 10 or len(row["candidate_ids"]) != 100:
            continue
        key = row["user_id"]
        query = {"query_id": row["query_id"], "history": [
            {"item_id": h["item_id"], "rating": h["rating"]} for h in row["history"]],
            "candidate_ids": row["candidate_ids"]}
        if key not in choices or digest(query) < digest(choices[key]):
            choices[key] = query
    selected = sorted(choices, key=lambda u: digest(["nextbook-serving-users", seed, u]))[:users]
    if len(selected) != users:
        raise ValueError("Insufficient distinct dev users with exact K10/N100 inputs")
    requests = []
    for user in selected:
        query = choices[user]
        order = list(range(100))
        random.Random(digest(["nextbook-serving-candidates", seed, query["query_id"]])).shuffle(order)
        for n in candidate_counts:
            indices = sorted(order[:n])
            copied = {**query, "candidate_ids": [query["candidate_ids"][i] for i in indices]}
            requests.append({"request_id": digest(copied), "query": copied,
                             # Accounting only; the serving API receives query.
                             "episodes": list(copied["candidate_ids"])})
    payload = {"protocol": PROTOCOL, "dev_sha256": hashlib.sha256(raw).hexdigest(),
               "users": users, "history": 10, "candidate_counts": list(candidate_counts),
               "seed": seed, "label_dependent_selection": False, "requests": requests}
    payload["workload_sha256"] = digest(payload)
    return payload


def verify_workload(value):
    if value.get("protocol") != PROTOCOL or value.get("workload_sha256") != digest(
            {k: v for k, v in value.items() if k != "workload_sha256"}):
        raise ValueError("Workload identity differs")
    rows = value["requests"]
    if len(rows) != value["users"] * len(value["candidate_counts"]):
        raise ValueError("Workload count differs")
    for row in rows:
        query = row["query"]
        if (row["request_id"] != digest(query) or row["episodes"] != query["candidate_ids"]
                or len(query["history"]) != value["history"]
                or len(query["candidate_ids"]) not in value["candidate_counts"]):
            raise ValueError("Workload request metadata differs")
    return value


def main(argv=None):
    """Write an immutable, label-free replay workload from prepared dev data."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--candidates", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    workload = build_workload(args.dev, users=args.users, candidate_counts=tuple(args.candidates), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(workload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"users": workload["users"], "requests": len(workload["requests"]),
                      "workload_sha256": workload["workload_sha256"]}))


if __name__ == "__main__":
    main()
