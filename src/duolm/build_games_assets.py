"""CPU-only, three-phase formal Games preparation: stage, contexts, materialize.

Stage derives first observations exclusively from the full raw 0-core reviews.
Contexts freezes the Books chronology before candidate availability checks and
exports only target-free training retrieval inputs. Materialize consumes actual
content retrieval and freezes mixed8 training, random100 dev and sealed test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from .build_next_item_assets import _clean, clip_shared_prefix, ENCODER_REPO, ENCODER_REVISION
from .games_data import (
    CandidateUnavailable, DEFAULT_CAPS, FIRST_SEEN_SOURCE, PREPARATION_SCHEMA,
    PROTOCOL, SELECTOR_NAMESPACE, SPLIT_NAMESPACE, CAP_NAMESPACE, TIME_FILTER,
    HistoryEntry, RecommendationContext, TimeAwareSampler, chronological_contexts,
    digest, first_seen_times, history_hash, load_timelines, manifest_identity,
    select_cohorts, select_temporal_queries, user_split, write_queries,
)
from .prepare_next_item import TOKENIZER_REPO, TOKENIZER_REVISION, file_hash

STAGE_SCHEMA = "games-assets-stage-v1"
CONTEXT_SCHEMA = "games-contexts-v1"
TOKEN_BUDGET = 512
TIMESTAMP_UNIT = "unix-milliseconds"
# Amazon reviews occur well after 1973. Seconds and microseconds cannot fit
# this broad millisecond envelope; do not infer a scale or convert values.
TIMESTAMP_MIN = 100_000_000_000
TIMESTAMP_MAX_EXCLUSIVE = 100_000_000_000_000


def _milliseconds(value, source):
    if isinstance(value, bool) or not isinstance(value, int) or not TIMESTAMP_MIN <= value < TIMESTAMP_MAX_EXCLUSIVE:
        raise ValueError(f"{source} timestamp must use Unix milliseconds; mixed seconds/microseconds are rejected")
    return value


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def _rows(path, values):
    with Path(path).open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


class _HashReader(io.RawIOBase):
    """Hash original compressed/uncompressed bytes during their one parse pass."""
    def __init__(self, stream):
        self.stream, self.hasher = stream, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        if count:
            self.hasher.update(memoryview(buffer)[:count])
        return count

    @property
    def sha256(self):
        return self.hasher.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@contextmanager
def _bound_text(path):
    """Stream JSONL/CSV while binding source bytes; detect concurrent changes."""
    path = Path(path)
    before = _stat(path)
    with path.open("rb") as raw:
        hashed = _HashReader(raw)
        buffered = io.BufferedReader(hashed)
        decompressed = gzip.GzipFile(fileobj=buffered) if path.suffix == ".gz" else buffered
        with io.TextIOWrapper(decompressed, encoding="utf-8", newline="") as stream:
            yield stream, hashed
            # Callers exhaust records. Reading a remaining tail would silently
            # bind unparsed data, so require the parser to reach real EOF.
            if stream.read(1):
                raise ValueError("source parser did not consume the complete frozen file")
    if _stat(path) != before:
        raise ValueError(f"input changed during Games preparation: {path.name}")


@contextmanager
def _fresh_output(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("frozen Games output must be fresh or explicitly resumed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        yield temporary
        if output.exists():
            output.rmdir()  # Only an empty user-requested destination is removed.
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_frozen_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"frozen asset checksum mismatch: {Path(path).name}")
    return json.loads(raw)


def _phase_manifest(root, name, schema):
    path = Path(root) / name
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != schema:
        raise ValueError("unsupported Games preparation phase schema")
    if report.get("identity_sha256") != manifest_identity(report):
        raise ValueError("Games phase identity mismatch")
    return report, hashlib.sha256(raw).hexdigest()


def _resume(output, name, schema, identity_inputs):
    report, _ = _phase_manifest(output, name, schema)
    if report.get("identity_inputs") != identity_inputs:
        raise ValueError("existing Games preparation differs from requested inputs/configuration")
    for filename, expected in report["files"].items():
        if file_hash(Path(output) / filename) != expected:
            raise ValueError(f"existing frozen Games checksum mismatch: {filename}")
    return report


def _finish(root, name, report, files, *, expected_files=None):
    report["files"] = {filename: file_hash(root / filename) for filename in files}
    if expected_files is not None and any(report["files"].get(filename) != sha for filename, sha in expected_files.items()):
        raise ValueError("shared Games assets changed while taking the frozen phase snapshot")
    report["identity_sha256"] = manifest_identity(report)
    _json(root / name, report)
    return report


def tokenizer_identity(snapshot_hashes=None):
    return {"b_tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
            "a_tokenizer": {"repo": ENCODER_REPO, "revision": ENCODER_REVISION},
            "tokenizer_files": snapshot_hashes or {}, "max_item_tokens": TOKEN_BUDGET,
            "truncation": "one-literal-source-prefix-fits-both-tokenizers-no-special-tokens"}


def canonical_metadata(row):
    """Title/Categories/Features/Description only; no ratings, price or reviews."""
    fields = {name: _clean(row.get(name)) for name in ("title", "categories", "features", "description")}
    if not fields["title"]:
        raise ValueError("Games metadata requires a nonempty title")
    text = "\n".join(f"{name.title()}: {value or '[' + name + ' missing]'}" for name, value in fields.items())
    return text, {"missing_" + name: int(not value) for name, value in fields.items()}


def build_stage(raw_reviews, ratings, metadata, output, *, b_tokenizer, a_tokenizer,
                tokenizer_files=None):
    """Freeze content and raw first observation; retain every distinct parent."""
    sources = {"raw0core_reviews_sha256": Path(raw_reviews), "ratings_sha256": Path(ratings),
               "metadata_sha256": Path(metadata)}
    parameters = {"protocol": PROTOCOL, "catalog_truncation": tokenizer_identity(tokenizer_files),
                  "code_sha256": {name: file_hash(Path(__file__).with_name(name))
                                  for name in ("build_games_assets.py", "games_data.py")}}
    if (Path(output) / "stage-manifest.json").exists():
        # Explicit recovery hashes each raw asset once; normal later phases use
        # only their frozen stage manifest and small derived assets.
        identity = {"sources": {key: file_hash(path) for key, path in sources.items()}, **parameters}
        return _resume(output, "stage-manifest.json", STAGE_SCHEMA, identity)
    audit, inventory, users = Counter(), set(), set()
    source_hashes = {}
    timestamp_ranges = {}
    csv_min, csv_max = None, None
    csv_first_seen = {}
    with _bound_text(ratings) as (stream, bound):
        reader = csv.DictReader(stream)
        if not {"user_id", "parent_asin", "rating", "timestamp"} <= set(reader.fieldnames or ()):
            raise ValueError("5-core CSV requires user_id,parent_asin,rating,timestamp")
        for row in reader:
            user, item = (row.get("user_id") or "").strip(), (row.get("parent_asin") or "").strip()
            if not user or not item:
                raise ValueError("5-core inventory has an unknown user/item identity")
            try:
                time = int(row["timestamp"])
            except (ValueError, TypeError) as error:
                raise ValueError("5-core CSV timestamp must be an integer in Unix milliseconds") from error
            _milliseconds(time, "5-core CSV")
            csv_first_seen[item] = min(time, csv_first_seen.get(item, time))
            csv_min, csv_max = min(time, csv_min if csv_min is not None else time), max(time, csv_max if csv_max is not None else time)
            users.add(user)
            inventory.add(item)
            audit["5core_rows"] += 1
    source_hashes["ratings_sha256"] = bound.sha256
    timestamp_ranges["ratings"] = {"field": "timestamp", "header": reader.fieldnames,
                                    "minimum": csv_min, "maximum": csv_max, "rows_checked": audit["5core_rows"]}
    first_seen = {}
    raw_min, raw_max = None, None
    with _bound_text(raw_reviews) as (stream, bound):
        for number, line in enumerate(stream, 1):
            audit["raw0core_review_rows"] += 1
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid raw 0-core JSON at line {number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"invalid raw 0-core object at line {number}")
            timestamp = _milliseconds(row.get("timestamp"), f"raw 0-core line {number}")
            raw_min, raw_max = min(timestamp, raw_min if raw_min is not None else timestamp), max(timestamp, raw_max if raw_max is not None else timestamp)
            item = row.get("parent_asin")
            if item not in inventory:
                continue
            first_seen[item] = min(timestamp, first_seen.get(item, timestamp))
            audit["raw0core_inventory_review_rows"] += 1
    source_hashes["raw0core_reviews_sha256"] = bound.sha256
    timestamp_ranges["raw0core"] = {"field": "timestamp", "minimum": raw_min, "maximum": raw_max,
                                    "rows_checked": audit["raw0core_review_rows"]}
    if any(first_seen[item] > csv_first_seen[item] for item in first_seen):
        raise ValueError("raw0core source is inconsistent: first observation occurs after a 5-core observation")
    items, structured, canonical_hashes = {}, {}, {}
    wanted = inventory & first_seen.keys()
    with _bound_text(metadata) as (stream, bound):
        for number, line in enumerate(stream, 1):
            audit["metadata_rows"] += 1
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid metadata JSON at line {number}") from error
            if not isinstance(row, dict) or row.get("parent_asin") not in wanted:
                continue
            item = row["parent_asin"]
            audit["inventory_metadata_rows"] += 1
            try:
                canonical, missing = canonical_metadata(row)
            except ValueError:
                audit["metadata_without_title"] += 1
                continue
            canonical_sha = hashlib.sha256(canonical.encode()).hexdigest()
            if item in canonical_hashes:
                if canonical_hashes[item] != canonical_sha:
                    raise ValueError(f"conflicting canonical Games metadata for parent_asin {item}")
                audit["identical_duplicate_parent_rows"] += 1
                continue
            text, b_count, a_count, truncated = clip_shared_prefix(
                canonical, b_tokenizer, a_tokenizer, limit=TOKEN_BUDGET)
            canonical_hashes[item] = canonical_sha
            items[item] = text
            audit.update(missing)
            audit["truncated_items"] += int(truncated)
            audit["b_content_tokens"] += b_count
            audit["a_content_tokens"] += a_count
            title = _clean(row.get("title")).lower()
            platform_hints = [platform for platform in ("ps5", "ps4", "xbox", "switch", "pc") if platform in title]
            accessory = any(word in title for word in ("controller", "headset", "charging", "cable", "adapter"))
            audit["accessory_title_hint_items"] += int(accessory)
            structured[item] = {"canonical_sha256": canonical_sha, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "b_tokens": b_count, "a_tokens": a_count, "truncated": truncated,
                                "missing_fields": [name.removeprefix("missing_") for name, value in missing.items() if value],
                                "platform_title_hints": platform_hints, "accessory_title_hint": accessory}
    source_hashes["metadata_sha256"] = bound.sha256
    if not items:
        raise ValueError("no 5-core items have usable metadata and a raw first observation")
    same_content = defaultdict(list)
    for item, canonical_sha in canonical_hashes.items():
        same_content[canonical_sha].append(item)
    audit.update(users=len(users), inventory_items=len(inventory), catalog_items=len(items),
                 inventory_missing_raw_observation=len(inventory - first_seen.keys()),
                 inventory_without_usable_catalog=len(inventory - items.keys()))
    records = {item: {"timestamp": first_seen[item], "source": FIRST_SEEN_SOURCE} for item in items}
    report = {"schema": STAGE_SCHEMA, "protocol": PROTOCOL, "identity_inputs": {"sources": source_hashes, **parameters},
              "sources": source_hashes, "catalog_truncation": parameters["catalog_truncation"], "audit": dict(audit),
              "timestamp_audit": {"unit": TIMESTAMP_UNIT, "conversion": "none; retain exact source integers",
                                  "accepted_range": [TIMESTAMP_MIN, TIMESTAMP_MAX_EXCLUSIVE],
                                  "range_upper_bound_exclusive": True, "sources": timestamp_ranges,
                                  "field_evidence": "official Video_Games 5-core CSV timestamp and raw0core JSONL timestamp both contain Unix milliseconds",
                                  "raw0core_covers_earliest_5core_time": "validated for every raw-covered inventory identity"},
              "first_observation": {"source": FIRST_SEEN_SOURCE, "computed_from": "all-raw-0core-reviews",
                                    "interpretation": "observed availability proxy, not verified release date"},
              "deduplication": "same parent_asin only; cross-parent titles/platforms retained",
              "metadata_fields": ["title", "categories", "features", "description"], "test_labels_accessed": False}
    with _fresh_output(output) as root:
        _json(root / "catalog.json", items)
        _json(root / "first-seen.json", records)
        _json(root / "inventory.json", sorted(inventory))
        _json(root / "identities.json", sorted(users))
        _json(root / "metadata-audit.json", {"items": structured,
              "identical_content_parent_groups": [sorted(group) for group in same_content.values() if len(group) > 1],
              "action": "audit only; no cross-parent item removed"})
        _finish(root, "stage-manifest.json", report,
                ("catalog.json", "first-seen.json", "inventory.json", "identities.json", "metadata-audit.json"))
    return report


def build_contexts(stage, ratings, output, *, seed=17):
    """Freeze fixed cohorts and original targets before real content retrieval."""
    if seed != 17 or isinstance(seed, bool):
        raise ValueError("formal first-round Games candidate seed is 17")
    stage = Path(stage)
    report, stage_sha = _phase_manifest(stage, "stage-manifest.json", STAGE_SCHEMA)
    items = _read_frozen_json(stage / "catalog.json", report["files"]["catalog.json"])
    records = _read_frozen_json(stage / "first-seen.json", report["files"]["first-seen.json"])
    users = _read_frozen_json(stage / "identities.json", report["files"]["identities.json"])
    identity_inputs = {"stage_manifest_sha256": stage_sha, "ratings_sha256": report["sources"]["ratings_sha256"],
                       "seed": seed, "query_cap": 5, "history_max": 10,
                       "code_sha256": {name: file_hash(Path(__file__).with_name(name))
                                       for name in ("build_games_assets.py", "games_data.py")}}
    if (Path(output) / "context-manifest.json").exists():
        if file_hash(ratings) != identity_inputs["ratings_sha256"]:
            raise ValueError("restored ratings differ from frozen Games stage")
        return _resume(output, "context-manifest.json", CONTEXT_SCHEMA, identity_inputs)
    # Dev/test selection is ID-only and occurs before availability measurement.
    cohorts = select_cohorts(users)
    allowed = set(user for split in ("train", "dev", "test") for user in cohorts[split])
    # Qualify the full train pool in the original cap order, then take the cap.
    # The stage inventory is only ~95K users; one CSV parse handles all candidates.
    allowed.update(user for user in users if user_split(user) == "train")
    with _bound_text(ratings) as (stream, bound):
        timelines, rejected, csv_audit = load_timelines(stream, allowed)
    if bound.sha256 != report["sources"]["ratings_sha256"]:
        raise ValueError("5-core CSV differs from the frozen Games stage")
    known, sampler = frozenset(items), TimeAwareSampler(records, seed=seed)
    contexts, chronology, unavailable = {split: [] for split in DEFAULT_CAPS}, {}, {"dev": {}, "test": {}}
    qualification = Counter()
    ordered_train = select_cohorts(users, caps={"train": len(users), "dev": 0, "test": 0}, selector_count=0)["train"]
    train_users = []
    for user in ordered_train:
        qualification["candidate_train_users_considered"] += 1
        if user in rejected:
            qualification[rejected[user]] += 1
            continue
        legal, audit = chronological_contexts(user, timelines.get(user, ()), split="train", known_item_ids=known)
        for name, value in audit.items():
            qualification["chronology_" + name] += value
        selected = select_temporal_queries(legal, limit=5) if legal else ()
        if not selected:
            qualification["no_legal_original_transition"] += 1
            continue
        try:
            for context in selected:
                if sampler.available_count(context) < 7:
                    raise CandidateUnavailable("selected_train_query_has_less_than_seven_time_legal_contrasts")
        except CandidateUnavailable as error:
            # Keep the quantile selection fixed. Never backfill an easier window.
            qualification[str(error)] += 1
            continue
        train_users.append(user)
        contexts["train"].extend(selected)
        qualification["qualified_train_users"] += 1
        if len(train_users) == DEFAULT_CAPS["train"]:
            break
    cohorts["train"] = train_users
    if not contexts["train"]:
        raise ValueError("no train users qualify under the unchanged Games chronology and time-legal mixed8")
    for split in ("dev", "test"):
        counts = Counter()
        for user in cohorts[split]:
            if user in rejected:
                unavailable[split][user] = rejected[user]
                continue
            legal, audit = chronological_contexts(user, timelines.get(user, ()), split=split, known_item_ids=known)
            counts.update(audit)
            if not legal:
                reasons = [name for name, value in audit.items() if value and name not in
                           {"time_groups", "positions_considered", "exact_duplicate_rows_collapsed", "legal_queries"}]
                unavailable[split][user] = "+".join(sorted(reasons)) or "no_legal_final_original_time_group"
                continue
            context = legal[0]
            try:
                if sampler.available_count(context) < 99:
                    raise CandidateUnavailable("final_target_has_less_than_99_time_legal_contrasts")
            except CandidateUnavailable as error:
                unavailable[split][user] = str(error)
                counts["insufficient_time_legal_candidate_pool"] += 1
                continue
            contexts[split].append(context)
        chronology[split] = dict(counts)
    chronology["train_qualification"] = dict(qualification)
    context_report = {"schema": CONTEXT_SCHEMA, "protocol": PROTOCOL, "identity_inputs": identity_inputs,
                      "sources": {**report["sources"], "catalog_sha256": report["files"]["catalog.json"]},
                      "catalog_truncation": report["catalog_truncation"], "stage_audit": report["audit"],
                      "timestamp_audit": report["timestamp_audit"],
                      "caps": DEFAULT_CAPS, "actual_cohorts": {key: len(value) for key, value in cohorts.items()},
                      "split_namespace": SPLIT_NAMESPACE, "cap_namespace": CAP_NAMESPACE,
                      "selector_namespace": SELECTOR_NAMESPACE, "query_cap": 5, "history_max": 10,
                      "seed": seed, "time_filter": TIME_FILTER, "chronology_audit": chronology,
                      "csv_audit": csv_audit, "unavailable_dev_users": sorted(unavailable["dev"]),
                      "unavailable_dev_reasons": unavailable["dev"], "unavailable_test_users": sorted(unavailable["test"]),
                      "unavailable_test_reasons": unavailable["test"], "test_labels_accessed": True,
                      "test_labels_sealed": True, "train_queries": len(contexts["train"]),
                      "dev_queries": len(contexts["dev"]), "test_queries": len(contexts["test"]),
                      "retrieval_input_scope": "train-only-target-free; dev/test random100 needs no retrieval"}
    with _fresh_output(output) as root:
        for name in ("catalog.json", "first-seen.json"):
            shutil.copyfile(stage / name, root / name)
        _json(root / "cohorts.json", cohorts)
        for split in DEFAULT_CAPS:
            filename = "sealed-test-contexts.jsonl" if split == "test" else split + "-contexts.jsonl"
            _rows(root / filename, (asdict(context) for context in contexts[split]))
        _rows(root / "retrieval-inputs.jsonl", (
            {"query_id": context.query_id, "user_id": context.user_id, "split": context.split,
             "history": [asdict(h) for h in context.history], "seen_item_ids": list(context.seen_item_ids),
             "cutoff_timestamp": context.target_timestamp, "history_hash": history_hash(context, items)}
            for context in contexts["train"]))
        _finish(root, "context-manifest.json", context_report,
                ("catalog.json", "first-seen.json", "cohorts.json", "train-contexts.jsonl", "dev-contexts.jsonl",
                 "sealed-test-contexts.jsonl", "retrieval-inputs.jsonl"),
                expected_files={name: report["files"][name] for name in ("catalog.json", "first-seen.json")})
    return context_report


def _read_contexts(path, expected, split):
    values, hasher, seen = [], hashlib.sha256(), set()
    with Path(path).open("rb") as stream:
        for line in stream:
            hasher.update(line)
            row = json.loads(line)
            if set(row) != set(RecommendationContext.__dataclass_fields__):
                raise ValueError("Games chronological context fields mismatch")
            row["history"] = tuple(HistoryEntry(**entry) for entry in row["history"])
            row["seen_item_ids"] = tuple(row["seen_item_ids"])
            context = RecommendationContext(**row)
            if context.split != split or user_split(context.user_id) != split or context.query_id in seen:
                raise ValueError("Games chronological context split/identity mismatch")
            seen.add(context.query_id)
            values.append(context)
    if hasher.hexdigest() != expected:
        raise ValueError("Games chronological context checksum mismatch")
    return values


def materialize(context_stage, retrieval, retrieval_manifest, output, *, seed=17):
    """Freeze actual retrieval mixed8 plus independent random100 dev/test."""
    if seed != 17 or isinstance(seed, bool):
        raise ValueError("formal first-round Games candidate seed is 17")
    root = Path(context_stage)
    report, context_sha = _phase_manifest(root, "context-manifest.json", CONTEXT_SCHEMA)
    raw_provenance = Path(retrieval_manifest).read_bytes()
    provenance = json.loads(raw_provenance)
    retrieval_manifest_sha = hashlib.sha256(raw_provenance).hexdigest()
    files = report["files"]
    if (provenance.get("schema") != "games-next-item-retrieval-v1" or provenance.get("history_only") is not True
            or provenance.get("time_filter") != TIME_FILTER or provenance.get("catalog_sha256") != files["catalog.json"]
            or provenance.get("first_seen_sha256") != files["first-seen.json"]
            or provenance.get("inputs_sha256") != files["retrieval-inputs.jsonl"]
            or not isinstance(provenance.get("retriever"), dict) or not provenance["retriever"]):
        raise ValueError("Games materialization requires real frozen time-filtered history-only retrieval")
    identity_inputs = {"context_manifest_sha256": context_sha, "retrieval_manifest_sha256": retrieval_manifest_sha,
                       "retrieval_results_sha256": provenance.get("results_sha256"), "seed": seed,
                       "code_sha256": {name: file_hash(Path(__file__).with_name(name))
                                       for name in ("build_games_assets.py", "games_data.py")}}
    if (Path(output) / "preparation-manifest.json").exists():
        if file_hash(retrieval) != provenance.get("results_sha256"):
            raise ValueError("restored retrieval results differ from the frozen manifest")
        return _resume(output, "preparation-manifest.json", PREPARATION_SCHEMA, identity_inputs)
    items = _read_frozen_json(root / "catalog.json", files["catalog.json"])
    records = _read_frozen_json(root / "first-seen.json", files["first-seen.json"])
    cohorts = _read_frozen_json(root / "cohorts.json", files["cohorts.json"])
    contexts = {split: _read_contexts(root / ("sealed-test-contexts.jsonl" if split == "test" else split + "-contexts.jsonl"),
                                      files["sealed-test-contexts.jsonl" if split == "test" else split + "-contexts.jsonl"], split)
                for split in DEFAULT_CAPS}
    by_id = {context.query_id: context for context in contexts["train"]}
    results, hasher = {}, hashlib.sha256()
    times = first_seen_times(records)
    with Path(retrieval).open("rb") as stream:
        for line in stream:
            hasher.update(line)
            row = json.loads(line)
            if set(row) != {"query_id", "history_hash", "retrieved_ids"}:
                raise ValueError("retrieval rows require query_id/history_hash/retrieved_ids")
            query_id = row["query_id"]
            if query_id not in by_id or query_id in results:
                raise ValueError("retrieval IDs must uniquely cover frozen train contexts")
            ids = row["retrieved_ids"]
            if (not isinstance(ids, list) or len(ids) > 200 or len(ids) != len(set(ids))
                    or any(not isinstance(item, str) or item not in items for item in ids)):
                raise ValueError("retrieval Top200 must have distinct catalog identities")
            context = by_id[query_id]
            if row["history_hash"] != history_hash(context, items):
                raise ValueError("retrieval history/content/cutoff hash mismatch")
            seen_items = set(context.seen_item_ids)
            if any(times[item] > context.target_timestamp or item in seen_items for item in ids):
                raise ValueError("retrieval must filter future/seen candidates before TopK")
            results[query_id] = ids
    if hasher.hexdigest() != provenance.get("results_sha256"):
        raise ValueError("retrieval results checksum mismatch")
    if set(results) != set(by_id):
        raise ValueError("retrieval must cover every frozen train context; no user/target deletion")
    sampler = TimeAwareSampler(records, seed=seed)
    train = [sampler.mixed_candidates(context, retrieved_ids=results[context.query_id]) for context in contexts["train"]]
    dev = [sampler.random_candidates(context, count=100) for context in contexts["dev"]]
    test = [sampler.random_candidates(context, count=100) for context in contexts["test"]]
    prepared = {key: value for key, value in report.items() if key not in {"schema", "files", "identity_inputs", "identity_sha256"}}
    prepared.update(schema=PREPARATION_SCHEMA, protocol=PROTOCOL, identity_inputs=identity_inputs,
                    retrieval=provenance, context_manifest_sha256=context_sha,
                    retrieval_manifest_sha256=retrieval_manifest_sha, train_policy="mixed-history",
                    candidate_policy="random-unobserved", train_candidates=8, dev_candidates=100, test_candidates=100,
                    seed=seed, train_queries=len(train), dev_queries=len(dev), test_queries=len(test),
                    sampling={"catalog_order": "first-observed-timestamp-then-parent-as-in-lexicographic-order",
                              "negative_sampling": "uniform-allowed-ranks-without-replacement",
                              "hard": "three-from-legal-history-top200", "random_train": 4,
                              "rng": "python-random-sha256-per-query-separated-hard-random-order",
                              "shuffle": "canonical-ID-sort-then-independent-shuffle"})
    with _fresh_output(output) as target:
        write_queries(target / "train.jsonl", train)
        write_queries(target / "dev.jsonl", dev)
        write_queries(target / "sealed-test.jsonl", test)
        for name in ("catalog.json", "first-seen.json", "cohorts.json"):
            shutil.copyfile(root / name, target / name)
        _finish(target, "preparation-manifest.json", prepared,
                ("train.jsonl", "dev.jsonl", "sealed-test.jsonl", "catalog.json", "first-seen.json", "cohorts.json"),
                expected_files={name: files[name] for name in ("catalog.json", "first-seen.json", "cohorts.json")})
    return prepared


def _load_tokenizers(a_path, b_path):
    """Verify only tokenizer assets, not multi-GB model weights, on CPU."""
    from transformers import AutoTokenizer
    result, proofs = {}, {}
    for name, path, repo, revision in (("a", a_path, ENCODER_REPO, ENCODER_REVISION),
                                       ("b", b_path, TOKENIZER_REPO, TOKENIZER_REVISION)):
        path = Path(path)
        manifest = json.loads((path / "snapshot-manifest.json").read_text())
        if manifest.get("repo_id") != repo or manifest.get("revision") != revision:
            raise ValueError("local tokenizer snapshot must bind the frozen public model revision")
        selected = {filename: record for filename, record in manifest.get("files", {}).items()
                    if filename in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"}}
        if not selected or "tokenizer_config.json" not in selected:
            raise ValueError("local snapshot does not bind tokenizer files")
        actual = {}
        for filename, record in selected.items():
            sha = file_hash(path / filename)
            if sha != record.get("sha256") or (path / filename).stat().st_size != record.get("bytes"):
                raise ValueError("tokenizer bytes differ from the pinned snapshot")
            actual[filename] = sha
        proofs[name] = actual
        result[name] = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        if not result[name].is_fast:
            raise ValueError("formal Games preparation requires fast tokenizer offsets")
    return result["b"], result["a"], proofs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--raw-reviews", "--raw0core", dest="raw_reviews", type=Path, required=True)
    stage.add_argument("--ratings", type=Path, required=True, help="official full Video_Games 5-core CSV")
    stage.add_argument("--metadata", type=Path, required=True)
    stage.add_argument("--a-tokenizer-path", type=Path, required=True)
    stage.add_argument("--b-tokenizer-path", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    context = sub.add_parser("contexts")
    context.add_argument("--stage", type=Path, required=True)
    context.add_argument("--ratings", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("materialize")
    final.add_argument("--contexts", "--stage", dest="contexts", type=Path, required=True)
    final.add_argument("--retrieval", type=Path, required=True)
    final.add_argument("--retrieval-manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "stage":
        b, a, proofs = _load_tokenizers(args.a_tokenizer_path, args.b_tokenizer_path)
        report = build_stage(args.raw_reviews, args.ratings, args.metadata, args.output,
                             b_tokenizer=b, a_tokenizer=a, tokenizer_files=proofs)
    elif args.phase == "contexts":
        report = build_contexts(args.stage, args.ratings, args.output)
    else:
        report = materialize(args.contexts, args.retrieval, args.retrieval_manifest, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key in {"schema", "identity_sha256", "audit", "actual_cohorts", "train_queries", "dev_queries", "test_queries"}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
