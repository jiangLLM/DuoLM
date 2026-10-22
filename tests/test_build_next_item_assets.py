"""Real-builder contracts on synthetic public rows, never existing test data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duolm.build_next_item_assets as assets
from duolm.build_next_item_assets import (
    MetadataConflictError, build_catalog, build_cohorts, canonical_metadata,
    clip_shared_prefix, qualify_train_users, scan_user_identities,
)
from duolm.next_item_data import user_split


class Tokenizer:
    def __init__(self, width=1):
        self.width = width

    def encode(self, text, **kwargs):
        return list(range(0, len(text), self.width))

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, min(i + self.width, len(text)))
                                   for i in range(0, len(text), self.width)]}


def users(split, count):
    result, index = [], 0
    while len(result) < count:
        user = "asset-user-" + str(index)
        if user_split(user) == split:
            result.append(user)
        index += 1
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user_id", "parent_asin", "rating", "timestamp"))
        writer.writerows(rows)


class CatalogTests(unittest.TestCase):
    def test_whitelist_keeps_missing_description_not_statistics(self):
        text, audit = canonical_metadata({"title": "<b>A &amp; B</b>", "author": {"name": "Writer", "link": "BADLINK"},
                                          "average_rating": "BADAVERAGE", "rating_number": "BADCOUNT",
                                          "review_text": "BADREVIEW"})
        self.assertIn("Title: A & B", text)
        self.assertIn("Author: Writer", text)
        self.assertIn("[description missing]", text)
        for excluded in ("BADLINK", "BADAVERAGE", "BADCOUNT", "BADREVIEW"):
            self.assertNotIn(excluded, text)
        self.assertEqual(audit["missing_description"], 1)
        with self.assertRaises(ValueError):
            canonical_metadata({"title": "<script>bad</script>"})

    def test_dual_budget_freezes_same_source_prefix(self):
        source = "长书标题 abc" * 30
        clipped, b, a, truncated = clip_shared_prefix(source, Tokenizer(2), Tokenizer(1), limit=24)
        self.assertTrue(source.startswith(clipped))
        self.assertTrue(truncated)
        self.assertLessEqual(b, 24)
        self.assertLessEqual(a, 24)
        self.assertEqual(clip_shared_prefix("short", Tokenizer(), Tokenizer(), limit=24), ("short", 5, 5, False))

    def test_stream_catalog_filter_duplicates_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "items.json"
            inventory.write_text(json.dumps(["a", "b", "missing"]))
            metadata = root / "metadata.jsonl"
            rows = [{"parent_asin": "a", "title": "A", "description": "x" * 100},
                    {"parent_asin": "a", "title": "A", "description": "x" * 100},
                    {"parent_asin": "b", "title": "B"},
                    {"parent_asin": "outside", "title": "Do not retain"}]
            metadata.write_text("not JSON\n" + "\n".join(map(json.dumps, rows)) + "\n")
            output = root / "assets"
            report = build_catalog(metadata, inventory, output, b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer(),
                                   max_item_tokens=80)
            catalog = json.loads((output / "catalog.json").read_text())
            self.assertEqual(set(catalog), {"a", "b"})
            self.assertIn("[description missing]", catalog["b"])
            self.assertEqual(report["audit"]["identical_canonical_parent_rows"], 1)
            self.assertEqual(report["audit"]["inventory_items_without_usable_metadata"], 1)
            self.assertEqual(report, build_catalog(metadata, inventory, output, b_tokenizer=Tokenizer(),
                                                   a_tokenizer=Tokenizer(), max_item_tokens=80))
            with self.assertRaisesRegex(ValueError, "differs"):
                build_catalog(metadata, inventory, output, b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer(),
                               max_item_tokens=79)

    def test_parent_conflict_is_audited_and_no_catalog_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, metadata = root / "ids.json", root / "meta.jsonl"
            inventory.write_text('["a"]')
            metadata.write_text(json.dumps({"parent_asin": "a", "title": "A"}) + "\n"
                                + json.dumps({"parent_asin": "a", "title": "Other"}) + "\n")
            with self.assertRaises(MetadataConflictError):
                build_catalog(metadata, inventory, root / "out", b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            self.assertFalse((root / "out/catalog.json").exists())
            failure = json.loads((root / "out/catalog-failure.json").read_text())
            self.assertEqual(failure["audit"]["conflicting_item_id"], "a")
            self.assertEqual(failure["catalog_published"], False)

    def test_inventory_and_its_manifest_are_bound_before_being_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, metadata, provenance = root / "ids.json", root / "meta.jsonl", root / "inventory.json"
            inventory.write_text('["a"]')
            metadata.write_text(json.dumps({"parent_asin": "a", "title": "A"}) + "\n")
            provenance.write_text(json.dumps({"uses_ratings": False, "uses_timestamps": False,
                                             "uses_user_association": False,
                                             "item_ids_sha256": assets.file_hash(inventory)}))
            for source in (inventory, provenance):
                with self.subTest(source=source.name):
                    original = source.read_bytes()
                    read_text = Path.read_text
                    changed = False

                    def read_then_change(path, *args, **kwargs):
                        nonlocal changed
                        value = read_text(path, *args, **kwargs)
                        if path == source and not changed:
                            # The returned old IDs/provenance must not acquire a new hash.
                            source.write_bytes(b'["b"]' if source == inventory else original + b"\n")
                            changed = True
                        return value

                    output = root / ("out-" + source.stem)
                    try:
                        with patch.object(Path, "read_text", read_then_change):
                            with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                                build_catalog(metadata, inventory, output, b_tokenizer=Tokenizer(),
                                              a_tokenizer=Tokenizer(), inventory_manifest=provenance)
                        self.assertTrue(changed)
                        self.assertFalse((output / "catalog-manifest.json").exists())
                    finally:
                        source.write_bytes(original)

    def test_metadata_changed_after_consumption_cannot_publish_or_resume_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, metadata = root / "ids.json", root / "meta.jsonl"
            inventory.write_text('["a"]')
            metadata.write_text(json.dumps({"parent_asin": "a", "title": "A"}) + "\n")
            original = metadata.read_bytes()
            output = root / "out"
            atomic_json = assets._atomic_json

            def publish_then_change(path, value):
                atomic_json(path, value)
                if Path(path).name == "catalog.json":
                    metadata.write_bytes(original + b"\n")

            with patch.object(assets, "_atomic_json", side_effect=publish_then_change):
                with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                    build_catalog(metadata, inventory, output, b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            self.assertFalse((output / "catalog-manifest.json").exists())
            metadata.write_bytes(original)
            frozen = root / "frozen"
            report = build_catalog(metadata, inventory, frozen, b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            self.assertEqual(report["identity"]["metadata_sha256"], assets.file_hash(metadata))
            hash_file = assets.file_hash

            def hash_then_change(path):
                value = hash_file(path)
                if Path(path) == frozen / "catalog.json":
                    metadata.write_bytes(original + b"\n")
                return value

            with patch.object(assets, "file_hash", side_effect=hash_then_change):
                with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                    build_catalog(metadata, inventory, frozen, b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            self.assertEqual(json.loads((frozen / "catalog-manifest.json").read_text()), report)


class CohortQualificationTests(unittest.TestCase):
    def test_cohort_sources_changed_after_qualification_cannot_publish_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, dev, test = users("train", 1), users("dev", 1), users("test", 1)
            ratings, catalog, existing = root / "raw.csv", root / "catalog.json", root / "existing.json"
            write_csv(ratings, [(train[0], "a", 5, 1), (train[0], "b", 1, 2)]
                      + [(user, "SEALED", "DO_NOT_PARSE", "DO_NOT_PARSE") for user in dev + test])
            catalog.write_text(json.dumps({item: item for item in ["a", "b", *[f"other-{i}" for i in range(12)]]}))
            existing.write_text(json.dumps({"train": train}))
            qualify = assets.qualify_train_users
            for source in (ratings, catalog, existing):
                with self.subTest(source=source.name):
                    original = source.read_bytes()
                    output = root / ("out-" + source.stem)

                    def qualify_then_change(*args, **kwargs):
                        result = qualify(*args, **kwargs)
                        source.write_bytes(original + b"\n")
                        return result

                    try:
                        with patch.object(assets, "qualify_train_users", side_effect=qualify_then_change):
                            with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                                build_cohorts(ratings, catalog, output, existing_cohorts=existing,
                                              train_users=1, dev_users=1, test_users=1, selector_users=1)
                        self.assertFalse((output / "cohort-manifest.json").exists())
                    finally:
                        source.write_bytes(original)

    def test_identity_scan_never_parses_sealed_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            all_users = users("train", 2) + users("dev", 2) + users("test", 2)
            write_csv(path, [(user, "", "NEVER_PARSE", "NEVER_PARSE") for user in all_users])
            found, audit = scan_user_identities(path)
            self.assertEqual(found, set(all_users))
            self.assertEqual(audit["parsed_columns"], ["user_id"])
            self.assertFalse(audit["ratings_parsed"])
            self.assertFalse(audit["timestamps_parsed"])

    def test_existing_train_retained_only_when_legal_and_dev_test_id_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, dev, test = users("train", 5), users("dev", 3), users("test", 3)
            rows = []
            for user in train:
                rows.extend([(user, "a", 5, 1), (user, "b", 1, 2), (user, "c", 2, 3)])
            for user in dev + test:
                rows.append((user, "SEALED", "DO_NOT_PARSE", "DO_NOT_PARSE"))
            # The first preferred user cannot be repaired by dropping this row.
            rows.append((train[0], "x", 3, "UNKNOWN_TIME"))
            raw = root / "raw.csv"
            write_csv(raw, rows)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({item: item for item in ["a", "b", "c", *[f"other-{i}" for i in range(12)]]}))
            existing = root / "existing.json"
            existing.write_text(json.dumps({"train": [train[0], train[1]]}))
            report = build_cohorts(raw, catalog, root / "out", existing_cohorts=existing, train_users=2,
                                   dev_users=2, test_users=2, selector_users=1, qualification_batch_users=2)
            frozen = json.loads((root / "out/cohorts.json").read_text())
            self.assertEqual(len(frozen["train"]), 2)
            self.assertIn(train[1], frozen["train"])
            self.assertNotIn(train[0], frozen["train"])
            self.assertTrue(set(frozen["checkpoint_selector"]) <= set(frozen["dev"]))
            self.assertFalse(report["test_labels_accessed"])
            self.assertEqual(report["qualification"]["counts"]["existing_qualified_users_retained"], 1)
            self.assertEqual(report, build_cohorts(raw, catalog, root / "out", existing_cohorts=existing,
                                                   train_users=2, dev_users=2, test_users=2, selector_users=1,
                                                   qualification_batch_users=2))

    def test_missing_history_does_not_manufacture_a_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            user = users("train", 1)[0]
            write_csv(path, [(user, "missing", 5, 1), (user, "b", 1, 2), (user, "c", 2, 3)])
            with self.assertRaisesRegex(ValueError, "only 0"):
                qualify_train_users(path, [user], {"b", "c", *[f"other-{i}" for i in range(10)]}, count=1)

    def test_low_rating_target_and_unknown_target_rating_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            user = users("train", 1)[0]
            write_csv(path, [(user, "a", 5, 1), (user, "b", 1, 2), (user, "c", "UNKNOWN_RATING", 3)])
            accepted, audit = qualify_train_users(path, [user], {"a", "b", "c", *[f"other-{i}" for i in range(10)]}, count=1)
            self.assertEqual(accepted, (user,))
            self.assertEqual(audit["selected_queries"], 2)

    def test_insufficient_candidate_capacity_or_cross_split_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            user = users("train", 1)[0]
            write_csv(path, [(user, "a", 5, 1), (user, "b", 1, 2)])
            with self.assertRaisesRegex(ValueError, "only 0"):
                qualify_train_users(path, [user], {"a", "b"}, count=1)
            with self.assertRaisesRegex(ValueError, "train only"):
                qualify_train_users(path, users("test", 1), {"a", "b"}, count=1)


if __name__ == "__main__":
    unittest.main()
