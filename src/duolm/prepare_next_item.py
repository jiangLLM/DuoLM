"""Freeze chronological train/dev groups from real, history-only retrieval.

Two phases keep targets out of the retriever's input file. This entry point
consumes a canonical item-text catalog and audited eligible user identities;
it does not fabricate missing metadata, retrieval results, or test labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from .next_item_data import (
    HistoryEntry, RecommendationContext, TrainingCandidateSampler,
    chronological_contexts, evaluation_candidates, load_timeline_csv,
    select_cohorts, select_temporal_queries, user_split, write_queries,
)

TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"
TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
PREPARATION_SCHEMA = "books-next-item-preparation-v1"


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verify_inputs_unchanged(paths, expected):
    """Do not publish provenance for files that changed while being consumed."""
    for name, path in paths.items():
        try:
            current = file_hash(path)
        except OSError as exc:
            raise ValueError(f"preparation input changed or became unreadable: {name}") from exc
        if current != expected[name]:
            raise ValueError(f"preparation input changed during preparation: {name}")


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")


def _fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("preparation output must be empty; frozen assets are never overwritten")
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_hash(context, items):
    """Bind visible content/feedback and all-past exclusion, never the target."""
    data = {"history": [(h.item_id, h.rating, h.timestamp, items[h.item_id])
                        for h in context.history], "seen_item_ids": context.seen_item_ids}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_cohorts(cohorts):
    for split in ("train", "dev", "test"):
        users = cohorts.get(split)
        if not isinstance(users, list) or not users or len(set(users)) != len(users):
            raise ValueError("cohorts require nonempty, unique user-ID arrays")
        if any(user_split(user) != split for user in users):
            raise ValueError("cohorts cross the original hash boundary")
    from .train_next_item import fixed_selector
    selector = cohorts.get("checkpoint_selector")
    if not isinstance(selector, list) or not selector or tuple(selector) != fixed_selector(cohorts["dev"], len(selector)):
        raise ValueError("selector must be the fixed hash-defined subset of dev")


def build_context_stage(ratings, catalog, cohorts, output, *, tokenizer,
                        tokenizer_identity, item_token_limit=512, query_cap=5):
    """Freeze transitions before retrieving; read selected train/dev labels only."""
    if not isinstance(item_token_limit, int) or isinstance(item_token_limit, bool) or item_token_limit < 1:
        raise ValueError("item_token_limit must be positive")
    _validate_cohorts(cohorts)
    source_paths = {"ratings_sha256": Path(ratings), "catalog_sha256": Path(catalog)}
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    raw_items = json.loads(Path(catalog).read_text(encoding="utf-8"))
    if not isinstance(raw_items, dict) or not raw_items:
        raise ValueError("catalog must map item IDs to canonical nonempty text")
    items, clipped = {}, 0
    for item, text in raw_items.items():
        if not isinstance(item, str) or not item or not isinstance(text, str) or not text.strip():
            raise ValueError("catalog identities/text must be nonempty strings")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("empty tokenized catalog item")
        if len(ids) > item_token_limit:
            clipped += 1
            ids = ids[:item_token_limit]
            text = tokenizer.decode(ids, skip_special_tokens=False)
            # A decoded prefix can retokenize differently at its final boundary.
            while len(tokenizer.encode(text, add_special_tokens=False)) > item_token_limit:
                ids = ids[:-1]
                text = tokenizer.decode(ids, skip_special_tokens=False)
        if not text.strip():
            raise ValueError("truncation produced empty item text")
        items[item] = text
    known = frozenset(items)
    contexts, unavailable, reasons, audits = {}, [], {}, {}
    for split in ("train", "dev"):
        timelines = load_timeline_csv(ratings, cohorts[split], split=split)
        selected, audit = [], Counter()
        for user in cohorts[split]:
            legal, counts = chronological_contexts(user, timelines[user], split=split, known_item_ids=known)
            audit.update(counts)
            if split == "train":
                if not legal:
                    raise ValueError(f"declared train user {user} has no legal metadata-complete transition")
                selected.extend(select_temporal_queries(legal, limit=query_cap))
            elif legal:
                selected.extend(legal)
            else:
                unavailable.append(user)
                reasons[user] = counts
        contexts[split], audits[split] = selected, dict(audit)
    output = _fresh_directory(output)
    _json(output / "catalog.json", items)
    _json(output / "cohorts.json", cohorts)
    for split, rows in contexts.items():
        _rows(output / (split + "-contexts.jsonl"), (asdict(context) for context in rows))
    _rows(output / "retrieval-inputs.jsonl", (
        {"query_id": context.query_id, "user_id": context.user_id, "split": context.split,
         "history": [asdict(entry) for entry in context.history],
         "seen_item_ids": list(context.seen_item_ids), "history_hash": history_hash(context, items)}
        for split in ("train", "dev") for context in contexts[split]))
    names = ("catalog.json", "cohorts.json", "train-contexts.jsonl", "dev-contexts.jsonl", "retrieval-inputs.jsonl")
    manifest = {"schema": "books-next-item-contexts-v1", "files": {name: file_hash(output / name) for name in names},
                "sources": source_hashes,
                "catalog_truncation": {"tokenizer": tokenizer_identity, "max_item_tokens": item_token_limit,
                                       "clipped_items": clipped}, "query_cap": query_cap,
                "unavailable_dev_users": unavailable, "unavailable_dev_reasons": reasons,
                "chronology_audit": audits, "test_labels_accessed": False}
    _verify_inputs_unchanged(source_paths, source_hashes)
    _json(output / "context-manifest.json", manifest)
    return manifest


def _contexts(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row["history"] = tuple(HistoryEntry(**event) for event in row["history"])
            row["seen_item_ids"] = tuple(row["seen_item_ids"])
            result.append(RecommendationContext(**row))
    return result


def assemble_groups(stage, retrieval, retrieval_manifest, output, *, seed=17):
    stage = Path(stage)
    required = {"catalog.json", "cohorts.json", "train-contexts.jsonl", "dev-contexts.jsonl", "retrieval-inputs.jsonl"}
    input_paths = {name: stage / name for name in sorted(required)}
    input_paths.update({"context_manifest_sha256": stage / "context-manifest.json",
                        "retrieval_manifest_sha256": Path(retrieval_manifest),
                        "retrieval_results_sha256": Path(retrieval)})
    input_hashes = {name: file_hash(path) for name, path in input_paths.items()}
    manifest = json.loads((stage / "context-manifest.json").read_text())
    if manifest.get("schema") != "books-next-item-contexts-v1" or manifest.get("test_labels_accessed") is not False:
        raise ValueError("invalid context-stage protocol")
    if set(manifest.get("files", {})) != required:
        raise ValueError("context manifest must bind every staged input")
    for name in required:
        if input_hashes[name] != manifest["files"][name]:
            raise ValueError("context stage checksum mismatch")
    provenance = json.loads(Path(retrieval_manifest).read_text())
    if (provenance.get("schema") != "books-next-item-retrieval-v1" or provenance.get("history_only") is not True
            or provenance.get("inputs_sha256") != manifest["files"]["retrieval-inputs.jsonl"]
            or provenance.get("catalog_sha256") != manifest["files"]["catalog.json"]
            or provenance.get("results_sha256") != input_hashes["retrieval_results_sha256"]
            or not isinstance(provenance.get("retriever"), dict) or not provenance["retriever"]):
        raise ValueError("retrieval must bind real results, history-only inputs, catalog and retriever identity")
    items = json.loads((stage / "catalog.json").read_text())
    contexts = {split: _contexts(stage / (split + "-contexts.jsonl")) for split in ("train", "dev")}
    by_id = {context.query_id: context for split in contexts.values() for context in split}
    if len(by_id) != sum(map(len, contexts.values())):
        raise ValueError("duplicate chronological context ID")
    results = {}
    with Path(retrieval).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"query_id", "history_hash", "retrieved_ids"}:
                raise ValueError("retrieval rows require query_id/history_hash/retrieved_ids")
            query_id = row["query_id"]
            if query_id not in by_id or query_id in results:
                raise ValueError("retrieval query IDs must be unique and exactly match frozen contexts")
            ids = row["retrieved_ids"]
            if not isinstance(ids, list) or any(not isinstance(item, str) or item not in items for item in ids):
                raise ValueError("retrieval output contains items outside the frozen catalog")
            if row["history_hash"] != history_hash(by_id[query_id], items):
                raise ValueError("retrieval history/content hash mismatch")
            results[query_id] = ids
    if set(results) != set(by_id):
        raise ValueError("retrieval results must cover every legal context; misses cannot delete users")
    sampler = TrainingCandidateSampler(items, seed=seed)
    train = [sampler.sample(context, retrieved_ids=results[context.query_id]) for context in contexts["train"]]
    dev = [evaluation_candidates(context, results[context.query_id], count=100) for context in contexts["dev"]]
    output = _fresh_directory(output)
    write_queries(output / "train.jsonl", train)
    if dev:
        write_queries(output / "dev.jsonl", dev)
    else:
        (output / "dev.jsonl").write_text("")
    for name in ("catalog.json", "cohorts.json"):
        shutil.copyfile(stage / name, output / name)
    prepared = {"schema": PREPARATION_SCHEMA,
                "files": {name: file_hash(output / name) for name in ("train.jsonl", "dev.jsonl", "catalog.json", "cohorts.json")},
                "context_manifest_sha256": input_hashes["context_manifest_sha256"],
                "retrieval_manifest_sha256": input_hashes["retrieval_manifest_sha256"], "retrieval": provenance,
                "sources": manifest["sources"], "catalog_truncation": manifest["catalog_truncation"],
                "unavailable_dev_users": manifest["unavailable_dev_users"],
                "unavailable_dev_reasons": manifest["unavailable_dev_reasons"],
                "chronology_audit": manifest["chronology_audit"], "seed": seed,
                "train_candidates": 8, "dev_candidates": 100, "query_cap": manifest["query_cap"],
                "test_labels_accessed": False, "train_queries": len(train), "dev_queries": len(dev)}
    _verify_inputs_unchanged(input_paths, input_hashes)
    _json(output / "preparation-manifest.json", prepared)
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    contexts = subparsers.add_parser("contexts", help="freeze chronology and target-free retrieval inputs")
    contexts.add_argument("--ratings", type=Path, required=True)
    contexts.add_argument("--catalog", type=Path, required=True, help="canonical item-ID to text JSON map")
    cohort_input = contexts.add_mutually_exclusive_group(required=True)
    cohort_input.add_argument("--cohorts", type=Path)
    cohort_input.add_argument("--eligible-user-ids", type=Path, help="audited eligible ID-only JSON array")
    contexts.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble", help="freeze complete groups from actual retrieval results")
    assemble.add_argument("--stage", type=Path, required=True)
    assemble.add_argument("--retrieval", type=Path, required=True)
    assemble.add_argument("--retrieval-manifest", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "contexts":
        if args.cohorts:
            cohorts = json.loads(args.cohorts.read_text())
            _validate_cohorts(cohorts)
            if {key: len(cohorts[key]) for key in ("train", "dev", "test", "checkpoint_selector")} != {
                    "train": 50000, "dev": 5000, "test": 10000, "checkpoint_selector": 2000}:
                raise ValueError("formal preparation requires 50K/5K/10K and fixed 2K selector")
        else:
            cohorts = {key: list(value) for key, value in asdict(select_cohorts(json.loads(args.eligible_user_ids.read_text()))).items()}
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
        report = build_context_stage(args.ratings, args.catalog, cohorts, args.output, tokenizer=tokenizer,
                                     tokenizer_identity={"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION})
    else:
        report = assemble_groups(args.stage, args.retrieval, args.retrieval_manifest, args.output)
    print(json.dumps({"output": str(args.output), "schema": report["schema"], "test_labels_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
