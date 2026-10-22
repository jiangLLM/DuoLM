from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from duolm.next_item_data import (
    HistoryEntry, RecommendationContext, RecommendationQuery, TrainingCandidateSampler,
    read_queries, select_cohorts, write_queries,
)
from duolm.candidate_data_validation import validate_candidate_preparation
import duolm.prepare_candidate_ranking as preparation
from duolm.prepare_next_item import file_hash


class CandidatePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "prepared-next-item"
        self.source.mkdir()
        selected = select_cohorts((f"candidate-user-{index}" for index in range(1000)),
                                  caps={"train": 24, "dev": 12, "test": 2}, selector_count=4)
        self.cohorts = {key: list(value) for key, value in asdict(selected).items()}
        self.items = {f"book-{index:03d}": f"Frozen prepared text {index}" for index in range(160)}
        self._json("catalog.json", self.items)
        self._json("cohorts.json", self.cohorts)
        self.queries = {"train": [], "dev": []}
        for split in self.queries:
            users = self.cohorts[split] if split == "train" else self.cohorts[split][:-1]
            for index, user in enumerate(users):
                candidates = tuple(f"book-{number:03d}" for number in range(20, 28))
                positive = 0
                if split == "dev":
                    # The old retriever missed alternate positives. Conversion
                    # preserves those queries and inserts their frozen targets.
                    start = 20 if index % 2 else 21
                    candidates = tuple(f"book-{number:03d}" for number in range(start, start + 100))
                    positive = 0 if start == 20 else None
                query = RecommendationQuery(
                    query_id=f"frozen-{split}-{index}", user_id=user, split=split,
                    history=tuple(HistoryEntry(f"book-{number:03d}", number % 5 + 1, number + 1)
                                  for number in range(5, 15)),
                    candidate_ids=candidates, positive_index=positive,
                    target_item_id="book-020", target_timestamp=16,
                    seen_item_ids=tuple(f"book-{number:03d}" for number in range(15)))
                self.queries[split].append(query)
            write_queries(self.source / (split + ".jsonl"), self.queries[split])
        self.manifest = {
            "schema": "books-next-item-preparation-v1",
            "files": {name: file_hash(self.source / name) for name in preparation.DATA_FILES},
            "sources": {"ratings_sha256": "1" * 64, "catalog_sha256": "2" * 64},
            "catalog_truncation": {"tokenizer": {"repo": "Qwen/Qwen3-0.6B-Base", "revision": "a" * 40},
                                   "max_item_tokens": 512, "clipped_items": 123},
            "unavailable_dev_users": self.cohorts["dev"][-1:],
            "unavailable_dev_reasons": {self.cohorts["dev"][-1]: {"missing_target_metadata": 1}},
            "chronology_audit": {"train": {"legal_queries": 24}, "dev": {"legal_queries": 11}},
            "seed": 17, "train_candidates": 8, "dev_candidates": 100, "query_cap": 5,
            "train_queries": 24, "dev_queries": 11, "test_labels_accessed": False,
            "context_manifest_sha256": "3" * 64, "retrieval_manifest_sha256": "4" * 64,
            "retrieval": {"schema": "books-next-item-retrieval-v1", "history_only": True,
                          "catalog_sha256": file_hash(self.source / "catalog.json"),
                          "inputs_sha256": "5" * 64, "results_sha256": "6" * 64,
                          "retriever": {"fixture": "old retriever is never invoked"}}}
        self._json("preparation-manifest.json", self.manifest)

    def _json(self, name, value):
        (self.source / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def _rebind(self):
        self.manifest["files"] = {name: file_hash(self.source / name) for name in preparation.DATA_FILES}
        self._json("preparation-manifest.json", self.manifest)

    def test_conversion_preserves_frozen_context_and_provenance(self):
        output = self.root / "candidate100"
        before = {name: file_hash(self.source / name) for name in preparation.SOURCE_FILES}
        report = preparation.prepare_candidate_ranking(self.source, output)
        self.assertEqual(report["schema"], "books-candidate-ranking-preparation-v1")
        self.assertEqual(report["protocol"], "books-candidate-ranking-v1")
        self.assertEqual(report["candidate_policy"], "random-unobserved")
        self.assertIs(report["target_included"], True)
        self.assertIs(report["test_labels_accessed"], False)
        self.assertEqual(report["source_preparation_manifest"], self.manifest)
        self.assertEqual(report["source_preparation_manifest_sha256"], before["preparation-manifest.json"])
        self.assertEqual(report["source_files_sha256"], before)
        self.assertEqual(report["identity_sha256"], preparation.manifest_identity(report))
        for key in ("sources", "catalog_truncation", "unavailable_dev_users", "unavailable_dev_reasons", "chronology_audit"):
            self.assertEqual(report[key], self.manifest[key])
        self.assertNotIn("retrieval", report)
        self.assertNotEqual(report["sources"]["catalog_sha256"], report["files"]["catalog.json"])
        for name, digest in report["files"].items():
            self.assertEqual(digest, file_hash(output / name))
        for name in ("catalog.json", "cohorts.json"):
            self.assertEqual((output / name).read_bytes(), (self.source / name).read_bytes())
        for split, count in (("train", 8), ("dev", 100)):
            converted = read_queries(output / (split + ".jsonl"), split=split)
            self.assertEqual(len(converted), len(self.queries[split]))
            for original, query in zip(self.queries[split], converted):
                self.assertEqual(replace(query, candidate_ids=original.candidate_ids,
                                         positive_index=original.positive_index), original)
                self.assertEqual(len(query.candidate_ids), count)
                self.assertEqual(len(set(query.candidate_ids)), count)
                self.assertEqual(query.candidate_ids.count(query.target_item_id), 1)
                self.assertEqual(query.candidate_ids[query.positive_index], query.target_item_id)
                # All 15 past items are excluded although only 10 are visible.
                self.assertTrue(set(query.seen_item_ids).isdisjoint(query.candidate_ids))
            self.assertGreater(len({query.positive_index for query in converted}), 3)
        self.assertEqual(before, {name: file_hash(self.source / name) for name in preparation.SOURCE_FILES})

    def test_reproducible_independent_of_global_rng_and_query_order(self):
        first, second = self.root / "first", self.root / "second"
        random.seed(800)
        state = random.getstate()
        report = preparation.prepare_candidate_ranking(self.source, first)
        self.assertEqual(random.getstate(), state)
        random.seed(2)
        for _ in range(100):
            random.random()
        second_report = preparation.prepare_candidate_ranking(self.source, second)
        self.assertEqual(report, second_report)
        for name in preparation.SOURCE_FILES:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        sampler = preparation.RandomCandidateSampler(reversed(self.items), seed=17)
        expected = {query.query_id: sampler.sample(query, count=8) for query in self.queries["train"]}
        shuffled = self.queries["train"][::-1]
        self.assertEqual(expected, {query.query_id: sampler.sample(query, count=8) for query in shuffled})

    def test_canonical_sort_separates_shuffle_from_positive_insertion_and_sampling_order(self):
        query = self.queries["train"][0]
        sampler = preparation.RandomCandidateSampler(self.items)
        original_rng = sampler._rng
        sampled = sampler.sample(query, count=8)
        shuffled_input = []

        class OrderRecorder:
            def shuffle(self, values):
                shuffled_input.extend(values)
                original_rng(query.query_id, "order").shuffle(values)

        with patch.object(sampler, "_rng", side_effect=lambda query_id, purpose:
                          OrderRecorder() if purpose == "order" else original_rng(query_id, purpose)):
            self.assertEqual(sampler.sample(query, count=8), sampled)
        self.assertEqual(shuffled_input, sorted(sampled.candidate_ids))

    def test_train100_is_configurable_while_dev_stays_fixed100(self):
        output8, output100 = self.root / "train8", self.root / "train100"
        report8 = preparation.prepare_candidate_ranking(self.source, output8)
        report100 = preparation.prepare_candidate_ranking(self.source, output100, train_candidates=100)
        self.assertTrue(all(len(query.candidate_ids) == 100 for query in read_queries(output100 / "train.jsonl", split="train")))
        self.assertEqual((output8 / "dev.jsonl").read_bytes(), (output100 / "dev.jsonl").read_bytes())
        self.assertNotEqual(report8["policy_sha256"], report100["policy_sha256"])
        self.assertNotEqual(report8["identity_sha256"], report100["identity_sha256"])
        for count in (True, 8.0, 7, 99):
            with self.assertRaisesRegex(ValueError, "train_candidates"):
                preparation.prepare_candidate_ranking(self.source, self.root / "invalid", train_candidates=count)

    def test_checksum_schema_and_duplicate_queries_are_rejected(self):
        output = self.root / "invalid"
        path = self.source / "catalog.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            preparation.prepare_candidate_ranking(self.source, output)
        self._rebind()
        self.manifest["schema"] = "books-candidate-ranking-preparation-v1"
        self._json("preparation-manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "schema"):
            preparation.prepare_candidate_ranking(self.source, output)
        self.manifest["schema"] = "books-next-item-preparation-v1"
        path = self.source / "dev.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[0])
        row["query_id"] = self.queries["train"][0].query_id
        rows[0] = json.dumps(row)
        path.write_text("\n".join(rows) + "\n")
        self._rebind()
        with self.assertRaisesRegex(ValueError, "unique across train/dev"):
            preparation.prepare_candidate_ranking(self.source, output)
        self.assertFalse(output.exists())

    def test_invalid_seen_or_duplicate_candidate_is_rejected(self):
        path = self.source / "train.jsonl"
        original = path.read_text()
        for kind in ("target_seen", "candidate_duplicate", "query_duplicate"):
            with self.subTest(kind=kind):
                rows = original.splitlines()
                row = json.loads(rows[0])
                if kind == "target_seen":
                    row["seen_item_ids"].append(row["target_item_id"])
                elif kind == "candidate_duplicate":
                    row["candidate_ids"][1] = row["candidate_ids"][0]
                else:
                    rows.append(rows[0])
                rows[0] = json.dumps(row)
                path.write_text("\n".join(rows) + "\n")
                self._rebind()
                with self.assertRaises(ValueError):
                    preparation.prepare_candidate_ranking(self.source, self.root / kind)
                self.assertFalse((self.root / kind).exists())

    def test_every_source_is_rechecked_before_success_is_published(self):
        for name in preparation.SOURCE_FILES:
            with self.subTest(source=name):
                source = self.source / name
                original = source.read_bytes()
                output = self.root / ("changed-" + name)
                original_write = preparation.write_queries
                changed = False

                def mutate_after_write(path, queries):
                    nonlocal changed
                    original_write(path, queries)
                    if not changed:
                        source.write_bytes(original + b"\n")
                        changed = True

                try:
                    with patch.object(preparation, "write_queries", side_effect=mutate_after_write):
                        with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                            preparation.prepare_candidate_ranking(self.source, output)
                    self.assertTrue(changed)
                    self.assertFalse((output / "preparation-manifest.json").exists())
                    self.assertFalse(output.exists())
                finally:
                    source.write_bytes(original)

    def test_no_test_labels_raw_ratings_or_stage_files_are_read(self):
        forbidden = {self.source / name for name in ("test_labels.jsonl", "test.jsonl", "ratings.csv", "context-manifest.json")}
        for path in forbidden:
            path.write_text("sealed invalid data")
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, forbidden)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            preparation.prepare_candidate_ranking(self.source, self.root / "no-test")

    def test_transient_live_mutation_cannot_change_parsed_history(self):
        source = self.source / "train.jsonl"
        original = source.read_bytes()
        rows = original.decode().splitlines()
        changed = json.loads(rows[0])
        changed["history"][0]["rating"] = 2
        rows[0] = json.dumps(changed)
        transient = ("\n".join(rows) + "\n").encode()
        real_read = preparation.read_queries

        def mutate_live_during_parse(path, **kwargs):
            source.write_bytes(transient)
            try:
                return real_read(path, **kwargs)
            finally:
                source.write_bytes(original)

        output = self.root / "transient-source"
        with patch.object(preparation, "read_queries", side_effect=mutate_live_during_parse):
            report = preparation.prepare_candidate_ranking(self.source, output)
        actual = read_queries(output / "train.jsonl", split="train")
        self.assertEqual(actual[0].history, self.queries["train"][0].history)
        self.assertEqual(report["source_files_sha256"]["train.jsonl"], file_hash(source))

    def test_transient_mutation_during_snapshot_copy_is_rejected(self):
        source = self.source / "catalog.json"
        original = source.read_bytes()
        real_copy = preparation.shutil.copyfile

        def mutate_copy(src, dest, *args, **kwargs):
            if Path(src) != source:
                return real_copy(src, dest, *args, **kwargs)
            source.write_bytes(original + b"\n")
            try:
                return real_copy(src, dest, *args, **kwargs)
            finally:
                source.write_bytes(original)

        output = self.root / "transient-copy"
        with patch.object(preparation.shutil, "copyfile", side_effect=mutate_copy):
            with self.assertRaisesRegex(ValueError, "source changed while taking"):
                preparation.prepare_candidate_ranking(self.source, output)
        self.assertFalse(output.exists())

    def test_mutation_during_staged_manifest_write_cannot_publish_success(self):
        output = self.root / "late-mutation"
        original_json = preparation._json
        source = self.source / "preparation-manifest.json"

        def mutate_after_manifest(path, value):
            original_json(path, value)
            source.write_text(source.read_text() + "\n")

        with patch.object(preparation, "_json", side_effect=mutate_after_manifest):
            with self.assertRaisesRegex(ValueError, "input changed during preparation"):
                preparation.prepare_candidate_ranking(self.source, output)
        self.assertFalse(output.exists())

    def test_identical_rerun_verifies_existing_and_different_identity_or_tamper_fails(self):
        output = self.root / "reusable"
        first = preparation.prepare_candidate_ranking(self.source, output)
        before = {name: (output / name).stat().st_mtime_ns for name in preparation.SOURCE_FILES}
        self.assertEqual(preparation.prepare_candidate_ranking(self.source, output), first)
        self.assertEqual(before, {name: (output / name).stat().st_mtime_ns for name in preparation.SOURCE_FILES})
        for kwargs in ({"seed": 18}, {"train_candidates": 100}):
            with self.assertRaisesRegex(ValueError, "identity"):
                preparation.prepare_candidate_ranking(self.source, output, **kwargs)
        (output / "train.jsonl").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            preparation.prepare_candidate_ranking(self.source, output)
        with self.assertRaisesRegex(ValueError, "overwrite its source"):
            preparation.prepare_candidate_ranking(self.source, self.source)
        incomplete = self.root / "partial"
        incomplete.mkdir()
        (incomplete / "train.jsonl").write_text("unfinished")
        with self.assertRaisesRegex(ValueError, "identical completed"):
            preparation.prepare_candidate_ranking(self.source, incomplete)

    def test_insufficient_catalog_fails_without_publishing_and_all_unavailable_dev_is_preserved(self):
        sampler = preparation.RandomCandidateSampler(self.items)
        query = self.queries["train"][0]
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            sampler.sample(query, count=146)
        self.manifest["unavailable_dev_users"] = self.cohorts["dev"]
        self.manifest["unavailable_dev_reasons"] = {user: {"missing_target_metadata": 1} for user in self.cohorts["dev"]}
        self.manifest["dev_queries"] = 0
        (self.source / "dev.jsonl").write_text("")
        self._rebind()
        output = self.root / "unavailable"
        report = preparation.prepare_candidate_ranking(self.source, output)
        self.assertEqual((output / "dev.jsonl").read_text(), "")
        self.assertEqual(report["unavailable_dev_users"], self.cohorts["dev"])
        self.assertEqual(report["unavailable_dev_reasons"], self.manifest["unavailable_dev_reasons"])
        self._json("catalog.json", {item: text for item, text in self.items.items() if item < "book-028"})
        self._rebind()
        insufficient = self.root / "insufficient"
        insufficient.mkdir()
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            preparation.prepare_candidate_ranking(self.source, insufficient, train_candidates=100)
        self.assertEqual(list(insufficient.iterdir()), [])

    def test_empty_output_directory_is_supported_and_duplicate_catalog_keys_fail(self):
        output = self.root / "empty"
        output.mkdir()
        preparation.prepare_candidate_ranking(self.source, output)
        self.assertTrue((output / "preparation-manifest.json").is_file())
        catalog = self.source / "catalog.json"
        catalog.write_text('{"book-000":"first","book-000":"duplicate"}')
        self._rebind()
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            preparation.prepare_candidate_ranking(self.source, self.root / "duplicate")

    def _mixed_reference(self):
        # Materialize real 1+4 random+3 history-Top200 groups using the original
        # sampler, rather than an eight-random stand-in.
        sampler = TrainingCandidateSampler(self.items, seed=17)
        train = []
        for query in self.queries["train"]:
            context = RecommendationContext(**{
                key: value for key, value in asdict(query).items()
                if key not in {"candidate_ids", "positive_index", "history"}}, history=query.history)
            train.append(sampler.sample(context, retrieved_ids=[f"book-{i:03d}" for i in range(30, 150)]))
        self.queries["train"] = train
        write_queries(self.source / "train.jsonl", train)
        self._rebind()
        reference = self.root / "evaluated-random100"
        preparation.prepare_candidate_ranking(self.source, reference, train_candidates=100)
        return reference

    def _split_config(self, *, train_policy="mixed-history", model_seed=29):
        return {"schema_version": preparation.PROTOCOL, "seed": model_seed, "candidate_seed": 17,
                "data": {"train_candidates": 8, "eval_candidates": 100, "candidate_policy": "random-unobserved",
                         "train_policy": train_policy, "history_max": 10, "query_cap": 5}}

    def _verify_split(self, output, *, config=None):
        train = read_queries(output / "train.jsonl", split="train")
        dev = read_queries(output / "dev.jsonl", split="dev")
        manifest = preparation._read_json(output / "preparation-manifest.json")
        hashes = {name: file_hash(output / name) for name in preparation.DATA_FILES}
        return validate_candidate_preparation(manifest, config or self._split_config(), hashes, train, dev)

    def test_mixed_train_and_evaluated_dev_preserve_exact_bytes_and_independent_seeds(self):
        reference = self._mixed_reference()
        before = {name: file_hash(reference / name) for name in preparation.SOURCE_FILES}
        output = self.root / "mixed"
        manifest = preparation.prepare_candidate_ranking(self.source, output, train_policy="mixed-history",
                                                          eval_source_dir=reference)
        self.assertEqual(manifest["schema"], preparation.SPLIT_PREPARATION_SCHEMA)
        self.assertEqual(manifest["train_policy"], "mixed-history")
        self.assertEqual(manifest["eval_policy"], "random-unobserved")
        self.assertEqual(manifest["train_sampling_provenance"]["retrieval"], self.manifest["retrieval"])
        self.assertEqual((output / "train.jsonl").read_bytes(), (self.source / "train.jsonl").read_bytes())
        self.assertEqual((output / "dev.jsonl").read_bytes(), (reference / "dev.jsonl").read_bytes())
        self.assertEqual(self._verify_split(output), manifest)
        self.assertEqual(before, {name: file_hash(reference / name) for name in preparation.SOURCE_FILES})
        self.assertEqual(preparation.prepare_candidate_ranking(self.source, output, train_policy="mixed-history",
                                                               eval_source_dir=reference), manifest)
        changed = self._split_config()
        changed["candidate_seed"] = 18
        with self.assertRaisesRegex(ValueError, "identity/policy"):
            self._verify_split(output, config=changed)
        changed = self._split_config()
        changed["data"]["train_policy"] = "random-unobserved"
        with self.assertRaisesRegex(ValueError, "identity/policy"):
            self._verify_split(output, config=changed)

    def test_mixed_is_explicit_and_requires_original_seed_and_retrieval_identity(self):
        with self.assertRaisesRegex(ValueError, "eval-source-dir"):
            preparation.prepare_candidate_ranking(self.source, self.root / "no-reference", train_policy="mixed-history")
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            preparation.prepare_candidate_ranking(self.source, self.root / "mixed100", train_policy="mixed-history",
                                                   train_candidates=100)
        reference = self._mixed_reference()
        with self.assertRaisesRegex(ValueError, "frozen source sampler seed"):
            preparation.prepare_candidate_ranking(self.source, self.root / "new-seed", train_policy="mixed-history",
                                                   seed=18, eval_source_dir=reference)
        self.manifest["retrieval"]["history_only"] = False
        self._json("preparation-manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "history-only retrieval"):
            preparation.prepare_candidate_ranking(self.source, self.root / "bad-retrieval", train_policy="mixed-history",
                                                   eval_source_dir=reference)

    def test_random_eight_with_reused_dev_stays_random_and_legacy_is_compatible(self):
        reference = self._mixed_reference()
        output = self.root / "random-control"
        manifest = preparation.prepare_candidate_ranking(self.source, output, eval_source_dir=reference)
        self.assertEqual(manifest["train_policy"], "random-unobserved")
        self.assertNotEqual((output / "train.jsonl").read_bytes(), (self.source / "train.jsonl").read_bytes())
        self.assertEqual((output / "dev.jsonl").read_bytes(), (reference / "dev.jsonl").read_bytes())
        self._verify_split(output, config=self._split_config(train_policy="random-unobserved"))
        legacy = self.root / "legacy"
        preparation.prepare_candidate_ranking(self.source, legacy)
        config = self._split_config(train_policy="random-unobserved", model_seed=17)
        del config["candidate_seed"]
        del config["data"]["train_policy"]
        self._verify_split(legacy, config=config)
        with self.assertRaisesRegex(ValueError, "identity/policy"):
            self._verify_split(legacy, config=self._split_config())

    def test_eval_reference_rejects_wrong_history_order_target_timestamp_candidates_and_cohort(self):
        reference = self._mixed_reference()
        original = {name: (reference / name).read_bytes() for name in preparation.SOURCE_FILES}
        dev = read_queries(reference / "dev.jsonl", split="dev")
        mutations = {
            "history": [replace(dev[0], history=(replace(dev[0].history[0], rating=2), *dev[0].history[1:])), *dev[1:]],
            "timestamp": [replace(dev[0], target_timestamp=17), *dev[1:]],
            "target": [replace(dev[0], target_item_id=dev[0].candidate_ids[(dev[0].positive_index + 1) % 100],
                               positive_index=(dev[0].positive_index + 1) % 100), *dev[1:]],
            "query-order": list(reversed(dev)),
            "candidate-order": [replace(dev[0], candidate_ids=tuple(reversed(dev[0].candidate_ids)),
                                         positive_index=99-dev[0].positive_index), *dev[1:]],
        }
        for kind, rows in mutations.items():
            with self.subTest(kind=kind):
                for name, data in original.items():
                    (reference / name).write_bytes(data)
                write_queries(reference / "dev.jsonl", rows)
                manifest = preparation._read_json(reference / "preparation-manifest.json")
                manifest["files"]["dev.jsonl"] = file_hash(reference / "dev.jsonl")
                preparation._json(reference / "preparation-manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "query order/history/target/timestamp|candidates/order"):
                    preparation.prepare_candidate_ranking(self.source, self.root / kind, train_policy="mixed-history",
                                                           eval_source_dir=reference)
        for name, data in original.items():
            (reference / name).write_bytes(data)
        (reference / "cohorts.json").write_bytes(original["cohorts.json"] + b"\n")
        manifest = preparation._read_json(reference / "preparation-manifest.json")
        manifest["files"]["cohorts.json"] = file_hash(reference / "cohorts.json")
        preparation._json(reference / "preparation-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "changed frozen source cohorts"):
            preparation.prepare_candidate_ranking(self.source, self.root / "changed-cohort", train_policy="mixed-history",
                                                   eval_source_dir=reference)

    def test_wrong_eval_source_identity_or_hash_cannot_publish_mixed_data(self):
        reference = self._mixed_reference()
        manifest = preparation._read_json(reference / "preparation-manifest.json")
        manifest["source_files_sha256"]["train.jsonl"] = "e" * 64
        manifest["identity_sha256"] = preparation.manifest_identity(manifest)
        preparation._json(reference / "preparation-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "source prepared file checksum"):
            preparation.prepare_candidate_ranking(self.source, self.root / "wrong-origin", train_policy="mixed-history",
                                                   eval_source_dir=reference)
        manifest["source_files_sha256"]["train.jsonl"] = self.manifest["files"]["train.jsonl"]
        manifest["identity_sha256"] = preparation.manifest_identity(manifest)
        preparation._json(reference / "preparation-manifest.json", manifest)
        (reference / "dev.jsonl").write_bytes((reference / "dev.jsonl").read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "complete frozen random-100"):
            preparation.prepare_candidate_ranking(self.source, self.root / "wrong-hash", train_policy="mixed-history",
                                                   eval_source_dir=reference)

    def test_every_split_source_is_rechecked_and_transient_eval_snapshot_changes_rejected(self):
        reference = self._mixed_reference()
        live = reference / "dev.jsonl"
        original = live.read_bytes()
        copy = preparation.shutil.copyfile
        def transient_copy(src, dst):
            if Path(src) != live:
                return copy(src, dst)
            live.write_bytes(original + b"\n")
            try:
                return copy(src, dst)
            finally:
                live.write_bytes(original)
        with patch.object(preparation.shutil, "copyfile", side_effect=transient_copy):
            with self.assertRaisesRegex(ValueError, "evaluation source changed while taking"):
                preparation.prepare_candidate_ranking(self.source, self.root / "transient-eval", train_policy="mixed-history",
                                                       eval_source_dir=reference)
        for source_dir in (self.source, reference):
            for name in preparation.SOURCE_FILES:
                live = source_dir / name
                original = live.read_bytes()
                output = self.root / (source_dir.name + "-" + name.replace(".", "-"))
                writer = preparation._json
                def mutate_after_manifest(path, value):
                    writer(path, value)
                    live.write_bytes(original + b"\n")
                try:
                    with patch.object(preparation, "_json", side_effect=mutate_after_manifest):
                        with self.assertRaisesRegex(ValueError, "input changed during"):
                            preparation.prepare_candidate_ranking(self.source, output, train_policy="mixed-history",
                                                                   eval_source_dir=reference)
                    self.assertFalse(output.exists())
                finally:
                    live.write_bytes(original)

    def test_split_loader_rejects_resigned_provenance_and_changed_preserved_bytes(self):
        reference = self._mixed_reference()
        output = self.root / "loader-audit"
        preparation.prepare_candidate_ranking(self.source, output, train_policy="mixed-history", eval_source_dir=reference)
        original = {name: (output / name).read_bytes() for name in preparation.SOURCE_FILES}
        manifest = preparation._read_json(output / "preparation-manifest.json")
        def train_seed(row):
            row["train_sampling_provenance"]["seed"] = 19
        def eval_seed(row):
            row["eval_sampling_provenance"]["seed"] = 19
        def source_hash(row):
            row["source_files_sha256"]["train.jsonl"] = "e" * 64
        def wrong_eval_hash(row):
            row["eval_source_files_sha256"]["dev.jsonl"] = "e" * 64
        for mutation in (train_seed, eval_seed, source_hash, wrong_eval_hash):
            row = json.loads(json.dumps(manifest))
            mutation(row)
            row["identity_sha256"] = preparation.manifest_identity(row)
            preparation._json(output / "preparation-manifest.json", row)
            with self.assertRaisesRegex(ValueError, "provenance|checksum|complete frozen"):
                self._verify_split(output)
        for split in ("train", "dev"):
            for name, data in original.items():
                (output / name).write_bytes(data)
            rows = read_queries(output / (split + ".jsonl"), split=split)
            write_queries(output / (split + ".jsonl"), reversed(rows))
            row = json.loads(json.dumps(manifest))
            row["files"][split + ".jsonl"] = file_hash(output / (split + ".jsonl"))
            row["identity_sha256"] = preparation.manifest_identity(row)
            preparation._json(output / "preparation-manifest.json", row)
            with self.assertRaisesRegex(ValueError, "training bytes/order|evaluation bytes"):
                self._verify_split(output)


if __name__ == "__main__":
    unittest.main()
