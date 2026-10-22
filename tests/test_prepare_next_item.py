import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duolm.prepare_next_item as preparation
from duolm.next_item_data import read_queries, select_cohorts, user_split
from duolm.prepare_next_item import (
    TOKENIZER_REPO, TOKENIZER_REVISION, assemble_groups, build_context_stage, file_hash,
)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        users = ["prepare-user-" + str(i) for i in range(500)]
        selected = select_cohorts(users, caps={"train": 1, "dev": 1, "test": 1}, selector_count=1)
        self.cohorts = {key: list(value) for key, value in asdict(selected).items()}
        self.catalog = self.root / "input-catalog.json"
        self.catalog.write_text(json.dumps({"book-" + str(i): "Book " + str(i) + " " + "x" * 50 for i in range(120)}))
        self.ratings = self.root / "ratings.csv"
        with self.ratings.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
            for split in ("train", "dev"):
                user = self.cohorts[split][0]
                writer.writerows([[user, "book-0", 5, 1], [user, "book-1", 1, 2], [user, "book-2", 2, 3]])
            # Malformed sealed labels must never be parsed, even during chronology.
            writer.writerow([self.cohorts["test"][0], "sealed", "not-a-rating", "not-a-timestamp"])
        self.stage = self.root / "stage"
        build_context_stage(self.ratings, self.catalog, self.cohorts, self.stage,
                            tokenizer=CharacterTokenizer(), tokenizer_identity={"fixture": True}, item_token_limit=32)

    def retrieval(self):
        rows = [json.loads(line) for line in (self.stage / "retrieval-inputs.jsonl").read_text().splitlines()]
        result = self.root / "retrieval.jsonl"
        result.write_text("\n".join(json.dumps({"query_id": row["query_id"], "history_hash": row["history_hash"],
                                              "retrieved_ids": ["book-" + str(i) for i in range(3, 120)]}) for row in rows) + "\n")
        manifest = self.root / "retrieval-manifest.json"
        manifest.write_text(json.dumps({"schema": "books-next-item-retrieval-v1", "history_only": True,
                                       "inputs_sha256": file_hash(self.stage / "retrieval-inputs.jsonl"),
                                       "catalog_sha256": file_hash(self.stage / "catalog.json"),
                                       "results_sha256": file_hash(result), "retriever": {"unit_fixture": True}}))
        return result, manifest

    def test_end_to_end_preparation_keeps_retrieval_misses_and_never_exports_targets_to_retriever(self):
        inputs = [json.loads(line) for line in (self.stage / "retrieval-inputs.jsonl").read_text().splitlines()]
        self.assertTrue(inputs)
        for row in inputs:
            self.assertNotIn("target_item_id", row)
            self.assertNotIn("target_rating", row)
        items = json.loads((self.stage / "catalog.json").read_text())
        self.assertTrue(all(len(text) <= 32 for text in items.values()))
        retrieval, manifest = self.retrieval()
        output = self.root / "prepared"
        prepared = assemble_groups(self.stage, retrieval, manifest, output)
        context_manifest = json.loads((self.stage / "context-manifest.json").read_text())
        self.assertEqual(prepared["sources"], {"ratings_sha256": file_hash(self.ratings),
                                              "catalog_sha256": file_hash(self.catalog)})
        self.assertEqual(prepared["context_manifest_sha256"], file_hash(self.stage / "context-manifest.json"))
        self.assertEqual(prepared["retrieval_manifest_sha256"], file_hash(manifest))
        self.assertEqual(context_manifest["sources"], prepared["sources"])
        train = read_queries(output / "train.jsonl", split="train")
        dev = read_queries(output / "dev.jsonl", split="dev")
        self.assertTrue(all(len(q.candidate_ids) == 8 for q in train))
        self.assertEqual(len(dev), 1)
        self.assertEqual(len(dev[0].candidate_ids), 100)
        self.assertIsNone(dev[0].positive_index)
        self.assertEqual(prepared["test_labels_accessed"], False)
        for name, digest in prepared["files"].items():
            self.assertEqual(digest, file_hash(output / name))

    def test_context_sources_changed_after_reading_cannot_publish_provenance(self):
        for name, source in (("catalog", self.catalog), ("ratings", self.ratings)):
            with self.subTest(source=name):
                original = source.read_bytes()
                original_hash = file_hash(source)
                output = self.root / ("changed-" + name)
                write_rows = preparation._rows
                changed = False

                def mutate_after_write(path, rows):
                    nonlocal changed
                    write_rows(path, rows)
                    if not changed:
                        # All source reads are complete when staged rows are written.
                        source.write_bytes(original + b"\n")
                        changed = True

                try:
                    with patch.object(preparation, "_rows", side_effect=mutate_after_write):
                        with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                            build_context_stage(self.ratings, self.catalog, self.cohorts, output,
                                                tokenizer=CharacterTokenizer(), tokenizer_identity={"fixture": True})
                    self.assertTrue(changed)
                    self.assertNotEqual(file_hash(source), original_hash)
                    self.assertFalse((output / "context-manifest.json").exists())
                finally:
                    source.write_bytes(original)

    def test_assembly_inputs_changed_after_reading_cannot_publish_provenance(self):
        retrieval, retrieval_manifest = self.retrieval()
        sources = [self.stage / name for name in (
            "catalog.json", "cohorts.json", "train-contexts.jsonl", "dev-contexts.jsonl",
            "retrieval-inputs.jsonl", "context-manifest.json")]
        sources.extend((retrieval, retrieval_manifest))
        for index, source in enumerate(sources):
            with self.subTest(source=source.name):
                original = source.read_bytes()
                output = self.root / ("changed-assembly-" + str(index))
                write_queries = preparation.write_queries
                changed = False

                def mutate_after_write(path, queries):
                    nonlocal changed
                    write_queries(path, queries)
                    if not changed:
                        # Results, manifests and contexts were consumed before group writing.
                        source.write_bytes(original + b"\n")
                        changed = True

                try:
                    with patch.object(preparation, "write_queries", side_effect=mutate_after_write):
                        with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                            assemble_groups(self.stage, retrieval, retrieval_manifest, output)
                    self.assertTrue(changed)
                    self.assertFalse((output / "preparation-manifest.json").exists())
                finally:
                    source.write_bytes(original)

    def test_declared_dev_remains_in_denominator_when_target_metadata_is_missing(self):
        catalog = json.loads(self.catalog.read_text())
        catalog.pop("book-2")
        self.catalog.write_text(json.dumps(catalog))
        # Train still has its earlier legal transition; dev cannot skip its last target.
        stage = self.root / "missing-stage"
        report = build_context_stage(self.ratings, self.catalog, self.cohorts, stage,
                                    tokenizer=CharacterTokenizer(), tokenizer_identity={"fixture": True})
        self.assertEqual(report["unavailable_dev_users"], self.cohorts["dev"])
        self.assertEqual((stage / "dev-contexts.jsonl").read_text(), "")

    def test_changed_stage_or_history_hash_fails_before_group_writing(self):
        retrieval, manifest = self.retrieval()
        rows = [json.loads(line) for line in retrieval.read_text().splitlines()]
        rows[0]["history_hash"] = "stale"
        retrieval.write_text("\n".join(map(json.dumps, rows)) + "\n")
        provenance = json.loads(manifest.read_text())
        provenance["results_sha256"] = file_hash(retrieval)
        manifest.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "history/content"):
            assemble_groups(self.stage, retrieval, manifest, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())
        (self.stage / "catalog.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            assemble_groups(self.stage, retrieval, manifest, self.root / "bad")

    def test_retrieval_provenance_and_full_coverage_are_required(self):
        retrieval, manifest = self.retrieval()
        provenance = json.loads(manifest.read_text())
        provenance["history_only"] = False
        manifest.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "history-only"):
            assemble_groups(self.stage, retrieval, manifest, self.root / "bad")
        provenance["history_only"] = True
        retrieval.write_text("")
        provenance["results_sha256"] = file_hash(retrieval)
        manifest.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "cover every"):
            assemble_groups(self.stage, retrieval, manifest, self.root / "bad")

    def test_stage_cannot_overwrite_frozen_assets_or_change_user_boundaries(self):
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            build_context_stage(self.ratings, self.catalog, self.cohorts, self.stage,
                                tokenizer=CharacterTokenizer(), tokenizer_identity={"fixture": True})
        wrong = dict(self.cohorts)
        wrong["train"] = self.cohorts["dev"]
        with self.assertRaisesRegex(ValueError, "boundary"):
            build_context_stage(self.ratings, self.catalog, wrong, self.root / "wrong",
                                tokenizer=CharacterTokenizer(), tokenizer_identity={"fixture": True})

    def test_real_preparation_artifacts_pass_formal_loader_and_semantic_changes_fail(self):
        from duolm.train_next_item import load_dataset
        self.stage = self.root / "formal-stage"
        build_context_stage(self.ratings, self.catalog, self.cohorts, self.stage,
                            tokenizer=CharacterTokenizer(),
                            tokenizer_identity={"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION})
        retrieval, provenance = self.retrieval()
        output = self.root / "formal-data"
        prepared = assemble_groups(self.stage, retrieval, provenance, output)
        cfg = {"data": {"train_users": 1, "dev_users": 1, "test_users": 1,
                        "selection_users": 1, "query_cap": 5, "history_max": 10}}
        train, dev, items, cohorts, hashes = load_dataset(output, cfg, formal=True)
        self.assertTrue(train)
        self.assertEqual(len(dev), 1)
        self.assertIsNone(dev[0].positive_index)
        self.assertEqual(len(cohorts["dev"]), 1)
        self.assertEqual(hashes["catalog"], file_hash(output / "catalog.json"))
        manifest_path = output / "preparation-manifest.json"
        for change in ({"test_labels_accessed": True}, {"query_cap": 999},
                       {"retrieval": {**prepared["retrieval"], "history_only": False}}):
            with self.subTest(change=change):
                manifest_path.write_text(json.dumps({**prepared, **change}))
                with self.assertRaises(ValueError):
                    load_dataset(output, cfg, formal=True)
        manifest_path.write_text(json.dumps(prepared))


if __name__ == "__main__":
    unittest.main()
