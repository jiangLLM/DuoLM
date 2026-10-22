"""Build real public-catalog and train-qualified next-item assets on CPU.

The catalog consumes an independent public item-ID inventory. The identity scan
only reads user IDs; only original-train users have ratings/times parsed during
qualification. Dev and test IDs are frozen before dev availability is measured.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile

from .next_item_data import (
    CAP_NAMESPACE, TimelineEvent, chronological_contexts, select_cohorts,
    select_temporal_queries, user_split,
)
from .prepare_next_item import (
    TOKENIZER_REPO, TOKENIZER_REVISION, _verify_inputs_unchanged, build_context_stage, file_hash,
)

ENCODER_REPO = "Qwen/Qwen3-Embedding-0.6B"
ENCODER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CATALOG_SCHEMA = "books-next-item-catalog-v1"
COHORT_SCHEMA = "books-next-item-qualified-cohorts-v1"


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _open_text(path):
    path = Path(path)
    return (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8", newline="")


def _clean(value):
    if isinstance(value, (list, tuple)):
        value = " ".join(_clean(part) for part in value)
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", " ", value,
                   flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<!--.*?-->|</?[A-Za-z][^>]*>", " ", value, flags=re.DOTALL)
    return re.sub(r"\s+", " ", value).strip()


def canonical_metadata(row):
    """Whitelist title/author/description; never stringify arbitrary objects."""
    title = _clean(row.get("title"))
    if not title:
        raise ValueError("metadata has no nonempty title")
    author = row.get("author")
    if isinstance(author, dict):
        author = author.get("name")
    elif isinstance(author, list):
        author = [part.get("name") if isinstance(part, dict) else part for part in author]
    author = _clean(author)
    description = _clean(row.get("description"))
    return (f"Title: {title}\nAuthor: {author or '[author missing]'}\n"
            f"Description: {description or '[description missing]'}"), {
                "missing_author": int(not author), "missing_description": int(not description)}


def clip_shared_prefix(text, b_tokenizer, a_tokenizer, *, limit=512):
    """Freeze one common text prefix that fits both actual tokenizer budgets."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("token limit must be a positive integer")
    b_ids = b_tokenizer.encode(text, add_special_tokens=False)
    a_ids = a_tokenizer.encode(text, add_special_tokens=False)
    if not b_ids or not a_ids:
        raise ValueError("metadata tokenizes to empty content")
    if len(b_ids) <= limit and len(a_ids) <= limit:
        return text, len(b_ids), len(a_ids), False
    # Prefer source-character offsets, avoiding a decoded partial UTF-8 token.
    try:
        offsets = b_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        end = offsets[min(limit, len(offsets)) - 1][1]
        candidate = text[:end]
    except (TypeError, KeyError, AttributeError):
        candidate = b_tokenizer.decode(b_ids[:limit], skip_special_tokens=False)
    # Prefix retokenization is not perfectly monotonic. Remove complete source
    # characters until both budgets are met; never change the source fields.
    while candidate:
        b_count = len(b_tokenizer.encode(candidate, add_special_tokens=False))
        a_count = len(a_tokenizer.encode(candidate, add_special_tokens=False))
        if 0 < b_count <= limit and 0 < a_count <= limit and candidate.strip():
            return candidate, b_count, a_count, True
        ratio = min(limit / max(b_count, 1), limit / max(a_count, 1))
        end = min(len(candidate) - 1, max(1, int(len(candidate) * ratio)))
        candidate = candidate[:end]
    raise ValueError("dual tokenizer truncation produced empty content")


def load_item_inventory(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value or value.strip() != value for value in values):
        raise ValueError("item inventory must be a nonempty ID-only JSON array")
    if len(set(values)) != len(values):
        raise ValueError("item inventory IDs must be unique")
    return frozenset(values)


class MetadataConflictError(ValueError):
    def __init__(self, item_id, audit):
        self.audit = dict(audit, conflicting_item_id=item_id)
        super().__init__(f"conflicting canonical metadata for parent_asin {item_id}")


def build_catalog(metadata, item_inventory, output, *, b_tokenizer, a_tokenizer,
                  max_item_tokens=512, inventory_manifest=None, tokenizer_snapshot_hashes=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {"metadata_sha256": Path(metadata), "item_inventory_sha256": Path(item_inventory)}
    if inventory_manifest is not None:
        source_paths["inventory_manifest_sha256"] = Path(inventory_manifest)
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    wanted = load_item_inventory(item_inventory)
    identity = {**source_hashes,
                "max_item_tokens": max_item_tokens,
                "b_tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
                "a_tokenizer": {"repo": ENCODER_REPO, "revision": ENCODER_REVISION},
                "tokenizer_snapshots": tokenizer_snapshot_hashes or {},
                "code_sha256": file_hash(__file__)}
    inventory_provenance = None
    if inventory_manifest is not None:
        inventory_provenance = json.loads(Path(inventory_manifest).read_text())
        if (inventory_provenance.get("uses_ratings") is not False
                or inventory_provenance.get("uses_timestamps") is not False
                or inventory_provenance.get("uses_user_association") is not False
                or inventory_provenance.get("item_ids_sha256") != identity["item_inventory_sha256"]):
            raise ValueError("public inventory must bind IDs without ratings, times or user association")
    completed = output / "catalog-manifest.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous.get("identity") != identity or previous.get("catalog_sha256") != file_hash(output / "catalog.json"):
            raise ValueError("existing frozen catalog differs from requested inputs/configuration")
        _verify_inputs_unchanged(source_paths, source_hashes)
        return previous
    if (output / "catalog.json").exists():
        raise ValueError("catalog without completed manifest requires explicit recovery, not overwrite")
    counts = Counter(inventory_items=len(wanted))
    items, content_hashes, seen = {}, {}, set()
    try:
        with _open_text(metadata) as stream:
            for line in stream:
                counts["metadata_rows_scanned"] += 1
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    counts["invalid_json"] += 1
                    continue
                if not isinstance(row, dict) or row.get("parent_asin") not in wanted:
                    continue
                item = row["parent_asin"]
                counts["inventory_metadata_rows"] += 1
                if item in seen:
                    counts["duplicate_parent_rows"] += 1
                seen.add(item)
                try:
                    text, missing = canonical_metadata(row)
                except ValueError:
                    counts["rows_without_nonempty_title"] += 1
                    continue
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                if item in content_hashes:
                    if content_hashes[item] != content_hash:
                        counts["conflicting_canonical_parent_rows"] += 1
                        raise MetadataConflictError(item, counts)
                    counts["identical_canonical_parent_rows"] += 1
                    continue
                clipped, b_count, a_count, truncated = clip_shared_prefix(
                    text, b_tokenizer, a_tokenizer, limit=max_item_tokens)
                items[item], content_hashes[item] = clipped, content_hash
                counts.update(missing)
                counts["clipped_items"] += int(truncated)
                counts["b_tokens"] += b_count
                counts["a_tokens"] += a_count
    except MetadataConflictError as error:
        _atomic_json(output / "catalog-failure.json", {"identity": identity, "audit": error.audit,
                                                      "catalog_published": False})
        raise
    if not items:
        raise ValueError("no usable metadata in the public inventory")
    counts["catalog_items"] = len(items)
    counts["inventory_items_without_usable_metadata"] = len(wanted) - len(items)
    _atomic_json(output / "catalog.json", items)
    report = {"schema": CATALOG_SCHEMA, "identity": identity, "audit": dict(counts),
              "metadata_fields": ["title", "author.name", "description"],
              "inventory": inventory_provenance, "catalog_sha256": file_hash(output / "catalog.json"),
              "test_labels_accessed": False}
    _verify_inputs_unchanged(source_paths, source_hashes)
    _atomic_json(completed, report)
    return report


def scan_user_identities(ratings):
    """Never inspect an item's identity, rating, or timestamp in this phase."""
    users, count = set(), 0
    with _open_text(ratings) as stream:
        reader = csv.DictReader(stream)
        if "user_id" not in (reader.fieldnames or []):
            raise ValueError("public CSV requires user_id")
        for row in reader:
            count += 1
            user = (row.get("user_id") or "").strip()
            if not user:
                raise ValueError("public CSV contains an unknown user identity")
            users.add(user)
    return users, {"public_rows_scanned": count, "users": len(users), "parsed_columns": ["user_id"],
                   "ratings_parsed": False, "timestamps_parsed": False}


def _train_batch(ratings, allowed):
    """Reject an invalid-time user's qualification, without editing chronology."""
    rows = {user: [] for user in allowed}
    rejected = set()
    if any(user_split(user) != "train" for user in allowed):
        raise ValueError("qualification may parse original-train labels only")
    with _open_text(ratings) as stream:
        reader = csv.DictReader(stream)
        if not {"user_id", "parent_asin", "rating", "timestamp"} <= set(reader.fieldnames or []):
            raise ValueError("rating CSV is missing required columns")
        for row in reader:
            user = (row.get("user_id") or "").strip()
            if user not in allowed or user in rejected:
                continue
            item = (row.get("parent_asin") or "").strip()
            try:
                timestamp = int(row["timestamp"])
                if not item or timestamp < 1:
                    raise ValueError
            except (ValueError, TypeError):
                rejected.add(user)
                continue
            try:
                rating_value = Decimal(row["rating"])
                rating = int(rating_value) if rating_value.is_finite() and rating_value in (1, 2, 3, 4, 5) else None
            except (ValueError, TypeError, InvalidOperation):
                rating = None
            rows[user].append(TimelineEvent(item, rating, timestamp))
    return rows, rejected


def qualify_train_users(ratings, user_ids, known_items, *, count=50_000, existing_users=(),
                        batch_users=100_000, query_cap=5):
    if count < 1 or batch_users < 1:
        raise ValueError("train count and qualification batch must be positive")
    known = known_items if isinstance(known_items, (set, frozenset)) else frozenset(known_items)
    pool = set(user_ids)
    if any(user_split(user) != "train" for user in pool):
        raise ValueError("qualification identity pool must be original train only")
    existing = set(existing_users)
    if any(user_split(user) != "train" for user in existing):
        raise ValueError("existing train IDs cross the original boundary")
    key = lambda user: (hashlib.sha256((CAP_NAMESPACE + "train|" + user).encode()).hexdigest(), user)
    preferred = sorted(existing & pool, key=key)
    ordered = preferred + sorted(pool - existing, key=key)
    accepted, audit = [], Counter(existing_requested=len(existing), existing_absent_from_public_ids=len(existing - pool))
    rejection_examples = {}
    query_counts = []
    for offset in range(0, len(ordered), batch_users):
        batch = ordered[offset:offset + batch_users]
        timelines, bad_users = _train_batch(ratings, set(batch))
        audit["raw_csv_passes"] += 1
        audit["candidate_batch_users_loaded"] += len(batch)
        for user in batch:
            audit["candidate_users_considered"] += 1
            if user in bad_users:
                reason = "unknown_identity_or_timestamp_user_rejected"
                legal = ()
            else:
                legal, chronology = chronological_contexts(user, timelines[user], split="train", known_item_ids=known)
                audit.update({"chronology_" + name: value for name, value in chronology.items()})
                reason = "no_legal_metadata_complete_transition"
            selected = select_temporal_queries(legal, limit=query_cap) if legal else ()
            if selected and any(len(known) - sum(item in known for item in context.seen_item_ids) - 1 < 7
                                for context in selected):
                selected, reason = (), "selected_query_has_less_than_seven_catalog_contrasts"
            if not selected:
                audit[reason] += 1
                if len(rejection_examples) < 100:
                    rejection_examples[user] = reason
                continue
            accepted.append(user)
            query_counts.append(len(selected))
            audit["qualified_users"] += 1
            audit["existing_qualified_users_retained"] += int(user in existing)
            if len(accepted) == count:
                return tuple(accepted), {"counts": dict(audit), "rejection_examples": rejection_examples,
                                         "selected_queries": sum(query_counts),
                                         "query_count_distribution": {str(key): value for key, value in Counter(query_counts).items()},
                                         "hard_pool_precheck": "catalog capacity only; actual Top200 validated after retrieval"}
    raise ValueError(f"only {len(accepted)} original-train users qualify; requested {count}; audit={dict(audit)}")


def build_cohorts(ratings, catalog, output, *, existing_cohorts=None, train_users=50_000,
                  dev_users=5_000, test_users=10_000, selector_users=2_000,
                  qualification_batch_users=100_000, query_cap=5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {"ratings_sha256": Path(ratings), "catalog_sha256": Path(catalog)}
    if existing_cohorts:
        source_paths["existing_cohorts_sha256"] = Path(existing_cohorts)
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    identity = {**source_hashes,
                "caps": {"train": train_users, "dev": dev_users, "test": test_users},
                "selector_users": selector_users, "query_cap": query_cap,
                "existing_cohorts_sha256": source_hashes.get("existing_cohorts_sha256"),
                "code_sha256": file_hash(__file__)}
    completed = output / "cohort-manifest.json"
    if completed.exists():
        report = json.loads(completed.read_text())
        if report.get("identity") != identity or report.get("cohorts_sha256") != file_hash(output / "cohorts.json"):
            raise ValueError("existing frozen cohorts differ from requested inputs/configuration")
        _verify_inputs_unchanged(source_paths, source_hashes)
        return report
    if (output / "cohorts.json").exists():
        raise ValueError("cohorts without completed manifest require explicit recovery")
    items = json.loads(Path(catalog).read_text())
    if not isinstance(items, dict) or not items:
        raise ValueError("catalog must contain real item texts")
    users, identity_audit = scan_user_identities(ratings)
    frozen = select_cohorts(users, caps={"train": 0, "dev": dev_users, "test": test_users},
                            selector_count=selector_users)
    train_pool = {user for user in users if user_split(user) == "train"}
    previous = json.loads(Path(existing_cohorts).read_text()) if existing_cohorts else {}
    existing = previous.get("train", [])
    if not isinstance(existing, list) or len(set(existing)) != len(existing):
        raise ValueError("existing train cohort must contain unique IDs")
    train, qualification = qualify_train_users(ratings, train_pool, frozenset(items), count=train_users,
                                               existing_users=existing, batch_users=qualification_batch_users,
                                               query_cap=query_cap)
    cohorts = {**asdict(frozen), "train": train}
    _atomic_json(output / "cohorts.json", cohorts)
    report = {"schema": COHORT_SCHEMA, "identity": identity, "identity_scan": identity_audit,
              "qualification": qualification, "cohorts_sha256": file_hash(output / "cohorts.json"),
              "test_ids_only": True, "test_labels_accessed": False,
              "dev_selection": "ID-only fixed hash; availability never changes its denominator"}
    _verify_inputs_unchanged(source_paths, source_hashes)
    _atomic_json(completed, report)
    return report


def verify_public_snapshot(path, *, repo, revision, manifest=None):
    """Verify the downloaded public revision before offline local loading."""
    path = Path(path).resolve()
    manifest = Path(manifest) if manifest else path / "snapshot-manifest.json"
    report = json.loads(manifest.read_text())
    if report.get("repo_id") != repo or report.get("revision") != revision:
        raise ValueError("local snapshot does not identify the required public repo/revision")
    files = report.get("files")
    if not isinstance(files, dict) or not files or "config.json" not in files:
        raise ValueError("public snapshot manifest requires a complete file hash inventory")
    for relative, record in files.items():
        filename = (path / relative).resolve()
        if (Path(relative).is_absolute() or path not in filename.parents or not filename.is_file()
                or not isinstance(record, dict) or filename.stat().st_size != record.get("bytes")
                or file_hash(filename) != record.get("sha256")):
            raise ValueError("local public snapshot file checksum/size/path differs from manifest")
    return file_hash(manifest)


def _tokenizers(cache_dir=None, b_tokenizer_path=None, a_tokenizer_path=None):
    from transformers import AutoTokenizer
    hashes = {}
    tokenizers = []
    for name, path, repo, revision in (("b", b_tokenizer_path, TOKENIZER_REPO, TOKENIZER_REVISION),
                                       ("a", a_tokenizer_path, ENCODER_REPO, ENCODER_REVISION)):
        if path:
            hashes[name] = verify_public_snapshot(path, repo=repo, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
        tokenizers.append(tokenizer)
    return *tokenizers, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    catalog = sub.add_parser("catalog")
    catalog.add_argument("--metadata", type=Path, required=True)
    catalog.add_argument("--catalog-item-ids", type=Path, required=True)
    catalog.add_argument("--inventory-manifest", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--cache-dir", type=Path)
    catalog.add_argument("--b-tokenizer-path", type=Path)
    catalog.add_argument("--a-tokenizer-path", type=Path)
    cohorts = sub.add_parser("cohorts")
    cohorts.add_argument("--ratings", type=Path, required=True)
    cohorts.add_argument("--catalog", type=Path, required=True)
    cohorts.add_argument("--output", type=Path, required=True)
    cohorts.add_argument("--existing-cohorts", type=Path)
    cohorts.add_argument("--qualification-batch-users", type=int, default=100_000)
    stage = sub.add_parser("stage")
    stage.add_argument("--ratings", type=Path, required=True)
    stage.add_argument("--assets", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--cache-dir", type=Path)
    stage.add_argument("--b-tokenizer-path", type=Path)
    stage.add_argument("--a-tokenizer-path", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "catalog":
        b, a, hashes = _tokenizers(args.cache_dir, args.b_tokenizer_path, args.a_tokenizer_path)
        report = build_catalog(args.metadata, args.catalog_item_ids, args.output, b_tokenizer=b, a_tokenizer=a,
                               inventory_manifest=args.inventory_manifest, tokenizer_snapshot_hashes=hashes)
    elif args.phase == "cohorts":
        report = build_cohorts(args.ratings, args.catalog, args.output, existing_cohorts=args.existing_cohorts,
                               qualification_batch_users=args.qualification_batch_users)
    else:
        catalog_manifest = json.loads((args.assets / "catalog-manifest.json").read_text())
        cohort_manifest = json.loads((args.assets / "cohort-manifest.json").read_text())
        if (catalog_manifest.get("catalog_sha256") != file_hash(args.assets / "catalog.json")
                or cohort_manifest.get("cohorts_sha256") != file_hash(args.assets / "cohorts.json")
                or cohort_manifest["identity"]["ratings_sha256"] != file_hash(args.ratings)
                or cohort_manifest["identity"]["catalog_sha256"] != catalog_manifest["catalog_sha256"]):
            raise ValueError("stage input hashes differ from frozen catalog/cohort assets")
        b, _, _ = _tokenizers(args.cache_dir, args.b_tokenizer_path, args.a_tokenizer_path)
        report = build_context_stage(args.ratings, args.assets / "catalog.json",
                                     json.loads((args.assets / "cohorts.json").read_text()), args.output,
                                     tokenizer=b, tokenizer_identity={"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION})
    print(json.dumps({"schema": report["schema"], "phase": args.phase, "output": str(args.output),
                      "test_labels_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
