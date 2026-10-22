"""Games protocol contracts on small synthetic CPU fixtures, never real labels."""
from dataclasses import asdict, replace
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duolm import games_data as data
from duolm import next_item_data as books
from duolm import build_games_assets as assets


TIMESTAMP_BASE = 1_600_000_000_000


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def users(split, count=1):
    values, index = [], 0
    while len(values) < count:
        user = "games-unit-user-" + str(index)
        if books.user_split(user) == split:
            values.append(user)
        index += 1
    return values


def context(split="train", target_timestamp=1000, seen=("past",), target="target"):
    return books.RecommendationContext("query-0", users(split)[0], split,
                                       (books.HistoryEntry(seen[-1], 1, target_timestamp - 1),),
                                       target, target_timestamp, tuple(sorted(seen)))


def records(values):
    return {item: {"timestamp": timestamp, "source": data.FIRST_SEEN_SOURCE} for item, timestamp in values.items()}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def build_fixture(root, *, unavailable_dev=False):
    """End-to-end stage/context/materialize fixture; usable by integration tests."""
    root = Path(root)
    raw, metadata, csv_path = root / "raw.jsonl.gz", root / "meta.jsonl", root / "ratings.csv"
    train_user, inventory_user = users("train", 2)
    dev_user, test_user = users("dev")[0], users("test")[0]
    ids = [f"game-{i:03d}" for i in range(140)] + ["future"]
    with gzip.open(raw, "wt") as stream:
        # These observations belong to users absent from the 5-core subset.
        for item in ids:
            stream.write(json.dumps({"user_id": "raw-only", "parent_asin": item,
                                     "timestamp": TIMESTAMP_BASE + (2000 if item == "future" else 1) * 1000}) + "\n")
        stream.write(json.dumps({"user_id": "raw-earlier", "parent_asin": ids[0], "timestamp": TIMESTAMP_BASE + 1000}) + "\n")
    write_jsonl(metadata, [{"parent_asin": item, "title": "Adventure " + item,
                           "categories": ["Video Games"], "features": ["Console edition"],
                           "description": ["Playable content"], "average_rating": "SECRETAVERAGE",
                           "price": "SECRETPRICE", "rating_number": "SECRETCOUNT"} for item in ids])
    rows = [[inventory_user, item, 3, 2000 if item == "future" else 50] for item in ids]
    rows += [[train_user, item, i % 5 + 1, 100 + i] for i, item in enumerate(ids[:14])]
    rows += [[dev_user, ids[0], 1, 100], [dev_user, "missing" if unavailable_dev else ids[1], 1, 101]]
    rows += [[test_user, ids[0], 5, 100], [test_user, ids[1], 1, 101]]
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user_id", "parent_asin", "rating", "timestamp"))
        writer.writerows([user, item, rating, TIMESTAMP_BASE + timestamp * 1000] for user, item, rating, timestamp in rows)
    stage, contexts, prepared = root / "stage", root / "contexts", root / "prepared"
    assets.build_stage(raw, csv_path, metadata, stage, b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
    manifest = assets.build_contexts(stage, csv_path, contexts)
    catalog = json.loads((contexts / "catalog.json").read_text())
    times = data.first_seen_times(json.loads((contexts / "first-seen.json").read_text()))
    input_rows = [json.loads(line) for line in (contexts / "retrieval-inputs.jsonl").read_text().splitlines()]
    retrieval, provenance_path = root / "retrieval.jsonl", root / "retrieval-manifest.json"
    result_rows = [{"query_id": row["query_id"], "history_hash": row["history_hash"],
                    "retrieved_ids": sorted(item for item in catalog if times[item] <= row["cutoff_timestamp"]
                                            and item not in row["seen_item_ids"])[:200]} for row in input_rows]
    write_jsonl(retrieval, result_rows)
    provenance = {"schema": "games-next-item-retrieval-v1", "history_only": True,
                  "time_filter": data.TIME_FILTER, "first_seen_sha256": manifest["files"]["first-seen.json"],
                  "catalog_sha256": manifest["files"]["catalog.json"],
                  "inputs_sha256": manifest["files"]["retrieval-inputs.jsonl"],
                  "results_sha256": assets.file_hash(retrieval),
                  "retriever": {"model": {"repo": assets.ENCODER_REPO, "revision": assets.ENCODER_REVISION},
                                "kind": "synthetic-unit-test-fixture"}}
    provenance_path.write_text(json.dumps(provenance))
    final = assets.materialize(contexts, retrieval, provenance_path, prepared)
    return {"root": root, "raw": raw, "ratings": csv_path, "metadata": metadata,
            "stage": stage, "contexts": contexts, "prepared": prepared,
            "retrieval": retrieval, "retrieval_manifest": provenance_path, "manifest": final}


class ChronologyTests(unittest.TestCase):
    def test_books_rule_equivalence_for_train_dev_and_quantile(self):
        events = [books.TimelineEvent(f"item-{i}", i % 5 + 1, 100 + i) for i in range(20)]
        variants = [events, events + [events[5]], events[:5] + [books.TimelineEvent("other", 2, 104)] + events[5:],
                    [replace(event, rating=None) if i == 7 else event for i, event in enumerate(events)]]
        for split in ("train", "dev"):
            for source in variants:
                for absent in (None, "item-19", "item-17"):
                    with self.subTest(split=split, absent=absent, source=len(source)):
                        known = {event.item_id for event in source} - ({absent} if absent else set())
                        a, aa = books.chronological_contexts(users(split)[0], source, split=split, known_item_ids=known)
                        b, ba = data.chronological_contexts(users(split)[0], source, split=split, known_item_ids=known)
                        self.assertEqual(aa, ba)
                        self.assertEqual([{k: v for k, v in asdict(row).items() if k != "query_id"} for row in a],
                                         [{k: v for k, v in asdict(row).items() if k != "query_id"} for row in b])
                        self.assertEqual([q.target_timestamp for q in books.select_temporal_queries(a)],
                                         [q.target_timestamp for q in data.select_temporal_queries(b)])

    def test_test_is_real_split_last_original_group_no_fallback(self):
        source = [books.TimelineEvent("a", 1, 1), books.TimelineEvent("b", 1, 2), books.TimelineEvent("missing", 1, 3)]
        result, audit = data.chronological_contexts(users("test")[0], source, split="test", known_item_ids={"a", "b"})
        self.assertFalse(result)
        self.assertEqual(audit["missing_target_metadata"], 1)
        valid, _ = data.chronological_contexts(users("test")[0], source[:2], split="test", known_item_ids={"a", "b"})
        sampler = data.TimeAwareSampler(records({"a": 1, "b": 1, "c": 1}))
        query = sampler.random_candidates(valid[0], count=2)
        self.assertEqual(query.split, "test")
        self.assertEqual(data.GamesQuery.from_dict(query.to_dict()), query)
        with self.assertRaisesRegex(ValueError, "test is sealed"):
            books.RecommendationQuery(**{k: v for k, v in asdict(query).items() if k != "history"}, history=query.history)

    def test_hash_boundary_and_caps_preserve_original_order(self):
        pool = sum([users(split, 4) for split in ("train", "dev", "test")], [])
        old = books.select_cohorts(pool, caps={"train": 4, "dev": 4, "test": 4}, selector_count=4)
        new = data.select_cohorts(pool, caps={"train": 10, "dev": 10, "test": 10}, selector_count=2000)
        self.assertEqual(asdict(old), {key: tuple(value) for key, value in new.items()})
        self.assertEqual(new, data.select_cohorts(reversed(pool), caps={"train": 10, "dev": 10, "test": 10}))


class SamplerTests(unittest.TestCase):
    def test_future_filter_past_only_and_boundary_inclusive(self):
        past = ("past", "older")
        mapping = {"past": 1, "older": 1, "target": 100, "future": 101,
                   **{f"legal-{i}": 100 for i in range(9)}}
        sampler = data.TimeAwareSampler(records(mapping))
        query = sampler.random_candidates(context(target_timestamp=100, seen=past), count=10)
        self.assertEqual(set(query.candidate_ids), {"target", *[f"legal-{i}" for i in range(9)]})
        self.assertEqual(len(query.candidate_ids), len(set(query.candidate_ids)))
        # A user's future-interaction identity is a legal unobserved contrast.
        self.assertIn("legal-0", query.candidate_ids)
        self.assertNotIn("future", query.candidate_ids)

    def test_mixed_hard_then_random_has_no_duplicates_and_is_deterministic(self):
        mapping = {"past": 1, "target": 1, "future": 2000, **{f"i-{i}": i + 1 for i in range(20)}}
        sampler = data.TimeAwareSampler(records(mapping))
        source = ["future", "past", "target", "i-0", "i-0", "i-1", "i-2"]
        a = sampler.mixed_candidates(context(), retrieved_ids=source)
        b = data.TimeAwareSampler(records(dict(reversed(list(mapping.items()))))).mixed_candidates(context(), retrieved_ids=source)
        self.assertEqual(a, b)
        self.assertEqual(len(a.candidate_ids), 8)
        self.assertEqual(len(set(a.candidate_ids)), 8)
        self.assertTrue({"i-0", "i-1", "i-2", "target"} <= set(a.candidate_ids))
        self.assertNotIn("future", a.candidate_ids)

    def test_allowed_rank_is_uniform_and_does_not_scan_catalog(self):
        sampler = data.TimeAwareSampler(records({"past": 1, "target": 1, **{f"i-{i}": 1 for i in range(4)}}))
        counts = {f"i-{i}": 0 for i in range(4)}
        for i in range(2000):
            sampled = sampler.random_candidates(replace(context(), query_id=f"q-{i}"), count=2)
            item = next(item for item in sampled.candidate_ids if item != "target")
            counts[item] += 1
        self.assertTrue(all(400 < count < 600 for count in counts.values()), counts)
        index_identity = id(sampler.indices)
        class NoIteration(tuple):
            def __iter__(self):
                raise AssertionError("per-query sampler iterated the full catalog")
        sampler.catalog = NoIteration(sampler.catalog)
        sampler.random_candidates(context(), count=3)
        self.assertEqual(id(sampler.indices), index_identity)

    def test_too_small_pool_does_not_change_target(self):
        sampler = data.TimeAwareSampler(records({"target": 1, "past": 1, "one": 1, "future": 2000}))
        with self.assertRaisesRegex(data.CandidateUnavailable, "insufficient_time_legal_pool"):
            sampler.random_candidates(context(), count=100)
        with self.assertRaisesRegex(data.CandidateUnavailable, "target_not_first_observed"):
            sampler.random_candidates(replace(context(), target_timestamp=1, history=(books.HistoryEntry("past", 1, 1),),
                                               target_item_id="future"), count=2)


class AssetTests(unittest.TestCase):
    def test_canonical_fields_and_cross_platform_versions_retained(self):
        text, audit = assets.canonical_metadata({"title": "<b>Series IV PS5</b>", "categories": ["Games"],
                         "features": ["Platform PS5"], "description": ["Action game"], "review_text": "SECRETREVIEW",
                         "rating": "SECRETRATING", "rating_number": "SECRETCOUNT", "price": "SECRETPRICE"})
        self.assertIn("Title: Series IV PS5", text)
        self.assertIn("Categories: Games", text)
        self.assertNotIn("SECRET", text)
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            stage = fixture["stage"]
            rows = [json.loads(line) for line in fixture["metadata"].read_text().splitlines()]
            rows[0]["title"], rows[1]["title"] = "Series IV PS5", "Series IV Xbox"
            write_jsonl(fixture["metadata"], rows)
            output = Path(directory) / "platform-stage"
            assets.build_stage(fixture["raw"], fixture["ratings"], fixture["metadata"], output,
                               b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            catalog = json.loads((output / "catalog.json").read_text())
            self.assertIn("PS5", catalog["game-000"])
            self.assertIn("Xbox", catalog["game-001"])
            self.assertEqual(len(catalog), len(json.loads((stage / "catalog.json").read_text())))

    def test_stage_rejects_mixed_timestamp_units_without_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            original_raw = fixture["raw"].read_bytes()
            with gzip.open(fixture["raw"], "rt") as stream:
                raw_rows = [json.loads(line) for line in stream]
            raw_rows[0]["timestamp"] //= 1000
            with gzip.open(fixture["raw"], "wt") as stream:
                for row in raw_rows:
                    stream.write(json.dumps(row) + "\n")
            output = Path(directory) / "bad-raw-stage"
            with self.assertRaisesRegex(ValueError, "Unix milliseconds"):
                assets.build_stage(fixture["raw"], fixture["ratings"], fixture["metadata"], output,
                                   b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            self.assertFalse(output.exists())
            fixture["raw"].write_bytes(original_raw)
            text = fixture["ratings"].read_text()
            fixture["ratings"].write_text(text.replace(str(TIMESTAMP_BASE + 50_000), str((TIMESTAMP_BASE + 50_000) // 1000), 1))
            with self.assertRaisesRegex(ValueError, "Unix milliseconds"):
                assets.build_stage(fixture["raw"], fixture["ratings"], fixture["metadata"], Path(directory) / "bad-csv-stage",
                                   b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())

    def test_raw_first_observation_is_not_5core_timestamp_and_hash_is_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            first = json.loads((fixture["stage"] / "first-seen.json").read_text())
            self.assertEqual(first["game-000"]["timestamp"], TIMESTAMP_BASE + 1000)  # 5-core earliest is 50.
            self.assertEqual(first["future"]["timestamp"], TIMESTAMP_BASE + 2_000_000)
            report = json.loads((fixture["stage"] / "stage-manifest.json").read_text())
            self.assertEqual(report["sources"]["raw0core_reviews_sha256"], assets.file_hash(fixture["raw"]))
            self.assertEqual(report["sources"]["ratings_sha256"], assets.file_hash(fixture["ratings"]))
            self.assertEqual(report["sources"]["metadata_sha256"], assets.file_hash(fixture["metadata"]))

    def test_end_to_end_training_never_opens_or_hashes_sealed_test(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            open_file, read_bytes = Path.open, Path.read_bytes
            def guarded_open(path, *args, **kwargs):
                self.assertNotIn("sealed-test", path.name)
                return open_file(path, *args, **kwargs)
            def guarded_read(path):
                self.assertNotIn("sealed-test", path.name)
                return read_bytes(path)
            with patch.object(Path, "open", guarded_open), patch.object(Path, "read_bytes", guarded_read):
                train, dev, items, cohorts, files = data.load_dataset(fixture["prepared"])
            self.assertEqual(len(train), 5)
            self.assertEqual(len(dev), 1)
            self.assertEqual(len(train[0].candidate_ids), 8)
            self.assertEqual(len(dev[0].candidate_ids), 100)
            self.assertNotIn("future", {i for query in (*train, *dev) for i in query.candidate_ids})
            self.assertTrue(all(query.split != "test" for query in (*train, *dev)))
            self.assertIn("first_seen", files)
            test, test_items, test_cohorts, manifest = data.load_sealed_test(
                fixture["prepared"], expected_identity_sha256=fixture["manifest"]["identity_sha256"])
            self.assertEqual(test[0].split, "test")
            self.assertEqual(len(test[0].candidate_ids), 100)
            self.assertEqual(items, test_items)
            with self.assertRaisesRegex(ValueError, "expected frozen"):
                data.load_sealed_test(fixture["prepared"], expected_identity_sha256="0" * 64)
            modified = dict(manifest, files={**manifest["files"], "sealed-test.jsonl": "f" * 64})
            self.assertNotEqual(data.manifest_identity(modified), manifest["identity_sha256"])

    def test_target_free_retrieval_hash_binds_cutoff_and_only_train(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            catalog = json.loads((fixture["contexts"] / "catalog.json").read_text())
            rows = [json.loads(line) for line in (fixture["contexts"] / "retrieval-inputs.jsonl").read_text().splitlines()]
            for row in rows:
                self.assertEqual(row["split"], "train")
                self.assertFalse(any("target" in name for name in row))
                self.assertEqual(row["history_hash"], data.history_hash(row, catalog))
                self.assertNotEqual(row["history_hash"], data.history_hash({**row, "cutoff_timestamp": row["cutoff_timestamp"] + 1}, catalog))

    def test_unavailable_fixed_dev_has_no_fallback_or_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory, unavailable_dev=True)
            train, dev, items, cohorts, files = data.load_dataset(fixture["prepared"])
            self.assertFalse(dev)
            self.assertEqual(cohorts["dev"], users("dev"))
            self.assertEqual(cohorts["unavailable_dev_users"], users("dev"))
            self.assertIn("missing_target_metadata", fixture["manifest"]["unavailable_dev_reasons"][users("dev")[0]])

    def test_frozen_output_resume_and_corruption_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            previous = assets.build_stage(fixture["raw"], fixture["ratings"], fixture["metadata"], fixture["stage"],
                                         b_tokenizer=Tokenizer(), a_tokenizer=Tokenizer())
            self.assertEqual(previous["schema"], assets.STAGE_SCHEMA)
            rerun = assets.materialize(fixture["contexts"], fixture["retrieval"], fixture["retrieval_manifest"], fixture["prepared"])
            self.assertEqual(rerun, fixture["manifest"])
            path = fixture["prepared"] / "train.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                data.load_dataset(fixture["prepared"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                assets.materialize(fixture["contexts"], fixture["retrieval"], fixture["retrieval_manifest"], fixture["prepared"])


if __name__ == "__main__":
    unittest.main()
