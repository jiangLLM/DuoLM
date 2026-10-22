"""Freeze train groups and random-100 dev from existing prepared Books data.

Only the five explicitly named train/dev preparation files are read. Existing
chronology, complete past-item exclusions, catalog text and cohort identities
remain frozen; neither a retriever nor raw ratings nor test labels are needed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from .next_item_data import SELECTOR_NAMESPACE, read_queries, user_split, write_queries
from .prepare_next_item import PREPARATION_SCHEMA as SOURCE_SCHEMA
from .prepare_next_item import _json, _verify_inputs_unchanged, file_hash


PROTOCOL = "books-candidate-ranking-v1"
PREPARATION_SCHEMA = "books-candidate-ranking-preparation-v1"
SPLIT_PREPARATION_SCHEMA = "books-candidate-ranking-preparation-v2"
TRAIN_POLICIES = ("random-unobserved", "mixed-history")
DATA_FILES = ("train.jsonl", "dev.jsonl", "catalog.json", "cohorts.json")
SOURCE_FILES = (*DATA_FILES, "preparation-manifest.json")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def candidate_policy(*, train_candidates=8, seed=17, train_policy="random-unobserved"):
    """Return the full, versioned sampling and ordering contract."""
    if isinstance(train_candidates, bool) or not isinstance(train_candidates, int) or train_candidates not in (8, 100):
        raise ValueError("train_candidates must be 8 or 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("candidate seed must be a nonnegative integer")
    if train_policy not in TRAIN_POLICIES:
        raise ValueError("unsupported training candidate policy")
    if train_policy == "mixed-history":
        if train_candidates != 8:
            raise ValueError("mixed-history preserves exactly eight training candidates")
        return {"protocol": PROTOCOL, "candidate_policy": train_policy,
                "target_included": True, "train_candidates": 8,
                "dev_candidates": 100, "seed": seed,
                "train_sampling": {
                    "sampler": "books-next-item-v1/TrainingCandidateSampler",
                    "random_unobserved": 4, "history_retrieval": 3,
                    "history_pool": "legal-deduplicated-history-retrieval-top-200",
                    "sampling": "uniform-without-replacement",
                    "candidate_order": "preserve-source-shuffled-candidate-order",
                    "materialization": "copy-frozen-source-train-jsonl"},
                "eval_sampling": candidate_policy(train_candidates=100, seed=seed)}
    return {"protocol": PROTOCOL, "candidate_policy": "random-unobserved",
            "target_included": True, "train_candidates": train_candidates,
            "dev_candidates": 100, "seed": seed,
            "catalog_order": "item-id-lexicographic",
            "exclude": "all-seen-item-ids-and-target",
            "sampling": "uniform-without-replacement",
            "rng": "python-random-sha256-per-query-separated-sample-and-order",
            "candidate_order": "canonical-item-id-sort-then-independent-shuffle"}


def manifest_identity(manifest):
    """Bind input bytes, exact candidate policy and all preparation code."""
    if manifest.get("schema") == SPLIT_PREPARATION_SCHEMA:
        # All source/evaluation manifests and the separate sampler contracts
        # are part of the identity. Output hashes are independently checked.
        return _digest({key: value for key, value in manifest.items()
                        if key not in {"identity_sha256", "files"}})
    return _digest({"schema": manifest["schema"],
                    "source_files_sha256": manifest["source_files_sha256"],
                    "policy_sha256": manifest["policy_sha256"],
                    "code_sha256": manifest["code_sha256"]})


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


class RandomCandidateSampler:
    """Index the catalog once; sample uniform allowed ranks without retries."""

    def __init__(self, catalog_ids, *, seed=17):
        candidate_policy(seed=seed)
        self.seed = seed
        self.catalog = tuple(sorted(catalog_ids))
        if len(set(self.catalog)) != len(self.catalog):
            raise ValueError("catalog identities must be unique")
        self.indices = {item: index for index, item in enumerate(self.catalog)}

    def _rng(self, query_id, purpose):
        return random.Random(int(_digest([PROTOCOL, self.seed, query_id, purpose]), 16))

    def sample(self, query, *, count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("candidate count must be an integer of at least two")
        if query.target_item_id not in self.indices:
            raise ValueError("target is absent from the frozen candidate catalog")
        excluded = set(query.seen_item_ids) | {query.target_item_id}
        blocked = sorted(self.indices[item] for item in excluded if item in self.indices)
        available = len(self.catalog) - len(blocked)
        if available < count - 1:
            raise ValueError(f"catalog cannot supply {count - 1} distinct unobserved items for {query.query_id}")
        blocked_ranks = tuple(index - position for position, index in enumerate(blocked))
        ranks = self._rng(query.query_id, "sample").sample(range(available), count - 1)
        candidates = [self.catalog[rank + bisect_right(blocked_ranks, rank)] for rank in ranks]
        # Sorting first makes the ordering independent of the target's insertion
        # position and of random.sample's ordering of the unobserved items.
        candidates = sorted([query.target_item_id, *candidates])
        self._rng(query.query_id, "order").shuffle(candidates)
        return replace(query, candidate_ids=tuple(candidates),
                       positive_index=candidates.index(query.target_item_id))


def _validate_source_manifest(manifest, hashes):
    if (not isinstance(manifest, dict) or manifest.get("schema") != SOURCE_SCHEMA
            or manifest.get("test_labels_accessed") is not False):
        raise ValueError("invalid source preparation schema or sealed-test protocol")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(DATA_FILES):
        raise ValueError("source manifest must bind exactly the four prepared data files")
    if any(manifest["files"][name] != hashes[name] for name in DATA_FILES):
        raise ValueError("source prepared file checksum mismatch")
    if manifest.get("train_candidates") != 8 or manifest.get("dev_candidates") != 100:
        raise ValueError("source candidate protocol must be the frozen 8/100 next-item task")
    cap = manifest.get("query_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("source query_cap must be a positive integer")
    sources = manifest.get("sources")
    if (not isinstance(sources, dict) or not {"ratings_sha256", "catalog_sha256"} <= sources.keys()
            or not all(_hex_digest(value) for value in sources.values())):
        raise ValueError("source preparation requires complete ratings/catalog source hashes")
    if not isinstance(manifest.get("catalog_truncation"), dict) or not manifest["catalog_truncation"]:
        raise ValueError("source preparation requires catalog_truncation provenance")
    if not isinstance(manifest.get("chronology_audit"), dict):
        raise ValueError("source preparation requires chronology_audit")


def _validate_mixed_source_manifest(manifest, hashes, *, seed):
    """Verify the old history-only mixed sampler before copying its groups."""
    _validate_source_manifest(manifest, hashes)
    if manifest.get("seed") != seed or isinstance(manifest.get("seed"), bool):
        raise ValueError("mixed-history candidate seed must equal the frozen source sampler seed")
    retrieval = manifest.get("retrieval")
    if (not isinstance(retrieval, dict) or retrieval.get("schema") != "books-next-item-retrieval-v1"
            or retrieval.get("history_only") is not True
            or retrieval.get("catalog_sha256") != hashes["catalog.json"]
            or not all(_hex_digest(retrieval.get(key))
                       for key in ("inputs_sha256", "catalog_sha256", "results_sha256"))
            or not isinstance(retrieval.get("retriever"), dict) or not retrieval["retriever"]
            or not all(_hex_digest(manifest.get(key))
                       for key in ("context_manifest_sha256", "retrieval_manifest_sha256"))):
        raise ValueError("mixed-history requires complete frozen history-only retrieval provenance")


def _validate_cohorts(cohorts):
    if not isinstance(cohorts, dict):
        raise ValueError("cohorts must be a mapping")
    for split in ("train", "dev", "test"):
        users = cohorts.get(split)
        if not isinstance(users, list) or not users or any(not isinstance(user, str) for user in users):
            raise ValueError("cohorts require nonempty user-ID arrays")
        if len(set(users)) != len(users) or any(user_split(user) != split for user in users):
            raise ValueError("cohorts require unique users within the original split boundary")
    selector = cohorts.get("checkpoint_selector")
    if not isinstance(selector, list) or not selector or len(selector) > len(cohorts["dev"]):
        raise ValueError("invalid fixed checkpoint selector")
    expected = sorted(cohorts["dev"], key=lambda user: (
        hashlib.sha256((SELECTOR_NAMESPACE + user).encode()).hexdigest(), user))[:len(selector)]
    if selector != expected:
        raise ValueError("checkpoint selector differs from the fixed hash-defined dev subset")


def _load_source(paths, hashes):
    manifest = _read_json(paths["preparation-manifest.json"])
    _validate_source_manifest(manifest, hashes)
    cohorts = _read_json(paths["cohorts.json"])
    _validate_cohorts(cohorts)
    items = _read_json(paths["catalog.json"])
    if not isinstance(items, dict) or not items or any(
            not isinstance(item, str) or not item or item.strip() != item
            or not isinstance(text, str) or not text.strip() for item, text in items.items()):
        raise ValueError("catalog must map unique item identities to nonempty text")
    unavailable = manifest.get("unavailable_dev_users")
    if (not isinstance(unavailable, list) or any(not isinstance(user, str) for user in unavailable)
            or len(set(unavailable)) != len(unavailable) or not set(unavailable) <= set(cohorts["dev"])):
        raise ValueError("invalid unavailable dev user declaration")
    if (not isinstance(manifest.get("unavailable_dev_reasons"), dict)
            or set(manifest["unavailable_dev_reasons"]) != set(unavailable)):
        raise ValueError("every unavailable dev user requires an explicit preparation reason")
    train = read_queries(paths["train.jsonl"], split="train")
    dev = (() if not paths["dev.jsonl"].stat().st_size and len(unavailable) == len(cohorts["dev"])
           else read_queries(paths["dev.jsonl"], split="dev"))
    query_ids = set()
    for split, queries in (("train", train), ("dev", dev)):
        for query in queries:
            if query.query_id in query_ids:
                raise ValueError("query IDs must be unique across train/dev")
            query_ids.add(query.query_id)
            if len(query.candidate_ids) != manifest[split + "_candidates"]:
                raise ValueError("source query candidate count differs from its manifest")
            required = {entry.item_id for entry in query.history} | set(query.candidate_ids) | {query.target_item_id}
            if not required <= items.keys():
                raise ValueError("source query references an item outside the prepared catalog")
    train_counts = Counter(query.user_id for query in train)
    dev_users = {query.user_id for query in dev}
    if (set(train_counts) != set(cohorts["train"])
            or max(train_counts.values()) > manifest["query_cap"]):
        raise ValueError("source training queries must preserve the declared cohort and query cap")
    if (dev_users & set(unavailable) or dev_users | set(unavailable) != set(cohorts["dev"])
            or len(dev) != len(cohorts["dev"]) - len(unavailable)):
        raise ValueError("source dev queries and unavailable users must preserve the complete declared cohort")
    if manifest.get("train_queries") != len(train) or manifest.get("dev_queries") != len(dev):
        raise ValueError("source manifest query counts do not match prepared data")
    return manifest, items, train, dev


def _verify_existing(output, expected):
    """Accept a complete identical preparation, never repair or overwrite it."""
    manifest_path = output / "preparation-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("output must be fresh or an identical completed preparation; frozen assets are never overwritten")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or {key: value for key, value in manifest.items() if key != "files"} != expected:
        raise ValueError("existing preparation identity or manifest differs; frozen assets are never overwritten")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(DATA_FILES):
        raise ValueError("existing preparation has an incomplete file manifest")
    for name in DATA_FILES:
        if not (output / name).is_file() or file_hash(output / name) != manifest["files"][name]:
            raise ValueError(f"existing preparation checksum mismatch: {name}")
    # Copied catalog and cohorts must remain the exact original frozen bytes.
    for name in ("catalog.json", "cohorts.json"):
        if manifest["files"][name] != expected["source_files_sha256"][name]:
            raise ValueError(f"existing preparation changed the frozen {name}")
    return manifest


def prepare_candidate_ranking(source_dir, output, *, train_candidates=8, seed=17,
                              train_policy="random-unobserved", eval_source_dir=None):
    """Create a separate, atomic fixed-candidate dataset or verify a rerun."""
    policy = candidate_policy(train_candidates=train_candidates, seed=seed, train_policy=train_policy)
    if train_policy == "mixed-history" and eval_source_dir is None:
        raise ValueError("mixed-history requires an existing random-100 --eval-source-dir")
    source_dir, output = Path(source_dir), Path(output)
    if source_dir.resolve() == output.resolve():
        raise ValueError("candidate preparation cannot overwrite its source directory")
    eval_source_dir = Path(eval_source_dir) if eval_source_dir is not None else None
    if eval_source_dir is not None and eval_source_dir.resolve() == output.resolve():
        raise ValueError("candidate preparation cannot overwrite its evaluation source directory")
    paths = {name: source_dir / name for name in SOURCE_FILES}
    # These five hashes precede all parsing. No directory discovery or raw/test
    # inputs can accidentally expand the set of authorized source files.
    hashes = {name: file_hash(path) for name, path in paths.items()}
    eval_paths = ({name: eval_source_dir / name for name in SOURCE_FILES}
                  if eval_source_dir is not None else None)
    eval_hashes = ({name: file_hash(path) for name, path in eval_paths.items()}
                   if eval_paths is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Parse only our owned snapshot. A live source may change and change back
    # between the initial and final checks; neither check binds a separate read.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.source-", dir=output.parent) as root:
        snapshots = {name: Path(root) / name for name in SOURCE_FILES}
        for name, path in paths.items():
            shutil.copyfile(path, snapshots[name])
            if file_hash(snapshots[name]) != hashes[name]:
                raise ValueError(f"source changed while taking preparation snapshot: {name}")
        if eval_paths is not None:
            eval_root = Path(root) / "evaluation"
            eval_root.mkdir()
            eval_snapshots = {name: eval_root / name for name in SOURCE_FILES}
            for name, path in eval_paths.items():
                shutil.copyfile(path, eval_snapshots[name])
                if file_hash(eval_snapshots[name]) != eval_hashes[name]:
                    raise ValueError(f"evaluation source changed while taking preparation snapshot: {name}")
            return _prepare_split_snapshot(
                snapshots, paths, hashes, eval_snapshots, eval_paths, eval_hashes,
                output, policy, train_candidates=train_candidates, seed=seed, train_policy=train_policy)
        return _prepare_snapshot(snapshots, paths, hashes, output, policy,
                                 train_candidates=train_candidates, seed=seed)


def _query_context(query):
    return {key: value for key, value in asdict(query).items()
            if key not in {"candidate_ids", "positive_index"}}


def _prepare_split_snapshot(paths, live_paths, hashes, eval_paths, live_eval_paths, eval_hashes,
                            output, policy, *, train_candidates, seed, train_policy):
    from .candidate_data_validation import validate_random_evaluation_reference

    source_manifest, items, train, original_dev = _load_source(paths, hashes)
    if train_policy == "mixed-history":
        _validate_mixed_source_manifest(source_manifest, hashes, seed=seed)
        if any(query.positive_index is None for query in train):
            raise ValueError("mixed-history source must include the target in every training group")
    reference = _read_json(eval_paths["preparation-manifest.json"])
    validate_random_evaluation_reference(reference, eval_hashes, seed=seed)
    if (reference.get("source_files_sha256") != hashes
            or reference.get("source_preparation_manifest") != source_manifest):
        raise ValueError("evaluation reference must originate from exactly the same frozen source")
    for name in ("catalog.json", "cohorts.json"):
        if eval_hashes[name] != hashes[name]:
            raise ValueError(f"evaluation reference changed frozen {name}")
    dev = (read_queries(eval_paths["dev.jsonl"], split="dev") if original_dev else ())
    if len(dev) != len(original_dev) or (not original_dev and eval_paths["dev.jsonl"].stat().st_size):
        raise ValueError("evaluation reference query cohort/count differs from the source")
    sampler = RandomCandidateSampler(items, seed=seed)
    for original, query in zip(original_dev, dev):
        if _query_context(original) != _query_context(query):
            raise ValueError("evaluation reference query order/history/target/timestamp differs from the source")
        if query != sampler.sample(original, count=100):
            raise ValueError("evaluation reference candidates/order differ from the frozen random sampler")
    code_paths = {name: Path(__file__).with_name(name) for name in (
        "prepare_candidate_ranking.py", "candidate_data_validation.py", "next_item_data.py", "prepare_next_item.py")}
    code_hashes = {name: file_hash(path) for name, path in code_paths.items()}
    train_provenance = {"policy": policy.get("train_sampling", policy),
                        "source_manifest_sha256": hashes["preparation-manifest.json"],
                        "source_train_sha256": hashes["train.jsonl"], "seed": seed}
    if train_policy == "mixed-history":
        train_provenance.update({"retrieval": source_manifest["retrieval"],
                                 "context_manifest_sha256": source_manifest["context_manifest_sha256"],
                                 "retrieval_manifest_sha256": source_manifest["retrieval_manifest_sha256"]})
    prepared = {"schema": SPLIT_PREPARATION_SCHEMA, "protocol": PROTOCOL,
                "candidate_policy": "random-unobserved", "train_policy": train_policy,
                "eval_policy": "random-unobserved", "target_included": True,
                "train_candidates": train_candidates, "dev_candidates": 100,
                "seed": seed, "candidate_seed": seed, "train_queries": len(train), "dev_queries": len(dev),
                "policy": policy, "policy_sha256": _digest(policy),
                "source_files_sha256": hashes, "source_preparation_manifest": source_manifest,
                "source_preparation_manifest_sha256": hashes["preparation-manifest.json"],
                "eval_source_files_sha256": eval_hashes, "eval_preparation_manifest": reference,
                "eval_preparation_manifest_sha256": eval_hashes["preparation-manifest.json"],
                "train_sampling_provenance": train_provenance,
                "eval_sampling_provenance": {
                    "policy": candidate_policy(train_candidates=100, seed=seed),
                    "reference_manifest_sha256": eval_hashes["preparation-manifest.json"],
                    "reference_identity_sha256": reference["identity_sha256"],
                    "reference_dev_sha256": eval_hashes["dev.jsonl"], "seed": seed},
                "code_files_sha256": code_hashes, "code_sha256": _digest(code_hashes),
                "test_labels_accessed": False}
    for key in ("sources", "catalog_truncation", "unavailable_dev_users", "unavailable_dev_reasons",
                "chronology_audit", "query_cap"):
        prepared[key] = source_manifest[key]
    prepared["identity_sha256"] = manifest_identity(prepared)
    def verify_inputs():
        _verify_inputs_unchanged(live_paths, hashes)
        _verify_inputs_unchanged(live_eval_paths, eval_hashes)
        _verify_inputs_unchanged(code_paths, code_hashes)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not output.is_dir():
            raise ValueError("preparation output must be a directory")
        existing = _verify_existing(output, prepared)
        if (existing["files"]["dev.jsonl"] != eval_hashes["dev.jsonl"]
                or (train_policy == "mixed-history" and existing["files"]["train.jsonl"] != hashes["train.jsonl"])):
            raise ValueError("existing preparation changed frozen train/evaluation bytes")
        verify_inputs()
        return existing
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        if train_policy == "mixed-history":
            shutil.copyfile(paths["train.jsonl"], temporary / "train.jsonl")
        else:
            write_queries(temporary / "train.jsonl", (sampler.sample(query, count=train_candidates) for query in train))
        shutil.copyfile(eval_paths["dev.jsonl"], temporary / "dev.jsonl")
        for name in ("catalog.json", "cohorts.json"):
            shutil.copyfile(paths[name], temporary / name)
        prepared["files"] = {name: file_hash(temporary / name) for name in DATA_FILES}
        _json(temporary / "preparation-manifest.json", prepared)
        verify_inputs()
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return prepared


def _prepare_snapshot(paths, live_paths, hashes, output, policy, *, train_candidates, seed):
    source_manifest, items, train, dev = _load_source(paths, hashes)
    code_paths = {name: Path(__file__).with_name(name) for name in (
        "prepare_candidate_ranking.py", "next_item_data.py", "prepare_next_item.py")}
    code_hashes = {name: file_hash(path) for name, path in code_paths.items()}
    prepared = {"schema": PREPARATION_SCHEMA, "protocol": PROTOCOL,
                "candidate_policy": "random-unobserved", "target_included": True,
                "train_candidates": train_candidates, "dev_candidates": 100, "seed": seed,
                "train_queries": len(train), "dev_queries": len(dev),
                "policy": policy, "policy_sha256": _digest(policy),
                "source_files_sha256": hashes,
                "source_preparation_manifest": source_manifest,
                "source_preparation_manifest_sha256": hashes["preparation-manifest.json"],
                "code_files_sha256": code_hashes, "code_sha256": _digest(code_hashes),
                "test_labels_accessed": False}
    for key in ("sources", "catalog_truncation", "unavailable_dev_users", "unavailable_dev_reasons",
                "chronology_audit", "query_cap"):
        prepared[key] = source_manifest[key]
    prepared["identity_sha256"] = manifest_identity(prepared)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not output.is_dir():
            raise ValueError("preparation output must be a directory")
        existing = _verify_existing(output, prepared)
        _verify_inputs_unchanged(live_paths, hashes)
        _verify_inputs_unchanged(code_paths, code_hashes)
        return existing
    sampler = RandomCandidateSampler(items, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_queries(temporary / "train.jsonl", (sampler.sample(query, count=train_candidates) for query in train))
        if dev:
            write_queries(temporary / "dev.jsonl", (sampler.sample(query, count=100) for query in dev))
        else:
            (temporary / "dev.jsonl").write_text("", encoding="utf-8")
        for name in ("catalog.json", "cohorts.json"):
            shutil.copyfile(paths[name], temporary / name)
        prepared["files"] = {name: file_hash(temporary / name) for name in DATA_FILES}
        _json(temporary / "preparation-manifest.json", prepared)
        # Everything is still unpublished. Verify after the last write so even
        # changes during manifest serialization cannot publish a stale success.
        _verify_inputs_unchanged(live_paths, hashes)
        _verify_inputs_unchanged(code_paths, code_hashes)
        # rename replaces an empty destination directory only, never an existing
        # nonempty preparation that appeared while this run was preparing data.
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-candidates", type=int, choices=(8, 100), default=8)
    parser.add_argument("--train-policy", choices=TRAIN_POLICIES, default="random-unobserved")
    parser.add_argument("--eval-source-dir", type=Path,
                        help="reuse exact dev bytes from an existing random-100 preparation (required for mixed-history)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    report = prepare_candidate_ranking(args.source_dir, args.output,
                                       train_candidates=args.train_candidates, seed=args.seed,
                                       train_policy=args.train_policy, eval_source_dir=args.eval_source_dir)
    print(json.dumps({"output": str(args.output), "schema": report["schema"],
                      "identity_sha256": report["identity_sha256"], "test_labels_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
