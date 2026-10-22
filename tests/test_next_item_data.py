"""Synthetic recommendation data tests; no existing data or test labels read."""
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from duolm import next_item_data as data


def users_for(split, count):
    users = []
    index = 0
    while len(users) < count:
        user = f"next-item-fixture-{index}"
        if data.user_split(user) == split:
            users.append(user)
        index += 1
    return users


def events(count=15):
    return tuple(data.TimelineEvent(f"book-{index}", index % 5 + 1, 100 + index)
                 for index in range(count))


def contexts(split="train", source=None, known=None, history_limit=10):
    source = events() if source is None else source
    known = {event.item_id for event in source} if known is None else known
    return data.chronological_contexts(users_for(split, 1)[0], source, split=split,
                                       known_item_ids=known, history_limit=history_limit)


def group(split="train"):
    context = contexts(split)[0][-1]
    if split == "train":
        return data.training_candidates(context,
                                        catalog_ids=[context.target_item_id, *[f"other-{i}" for i in range(10)]],
                                        retrieved_ids=[f"other-{i}" for i in range(3)])
    return data.evaluation_candidates(context, [context.target_item_id, "other-1"], count=2)


class CohortTests(unittest.TestCase):
    def test_original_hash_boundary_exact(self):
        for user in ["abc", "用户", "next-item-fixture-1"]:
            bucket = int(hashlib.sha256(("amz-split-20260927|" + user).encode()).hexdigest(), 16) % 10_000
            expected = "train" if bucket < 8000 else "dev" if bucket < 9000 else "test"
            self.assertEqual(data.user_split(user), expected)

    def test_nested_caps_order_stability_and_fixed_dev_selector(self):
        pool = sum([users_for(split, 20) for split in ("train", "dev", "test")], [])
        first = data.select_cohorts(pool, caps={"train": 8, "dev": 10, "test": 12}, selector_count=4)
        reversed_order = data.select_cohorts(reversed(pool), caps={"train": 8, "dev": 10, "test": 12},
                                           selector_count=4)
        second = data.select_cohorts(pool, caps={"train": 16, "dev": 10, "test": 12}, selector_count=4)
        self.assertEqual(first, reversed_order)
        self.assertEqual(first.train, second.train[:8])
        self.assertEqual(first.checkpoint_selector, second.checkpoint_selector)
        self.assertTrue(set(first.checkpoint_selector) <= set(first.dev))
        self.assertFalse(set(first.train) & set(first.dev))
        self.assertFalse((set(first.train) | set(first.dev)) & set(first.test))

    def test_defaults_are_requested_cohort_sizes(self):
        self.assertEqual(data.DEFAULT_CAPS, {"train": 50_000, "dev": 5_000, "test": 10_000})

    def test_insufficient_split_cannot_borrow_users(self):
        with self.assertRaisesRegex(ValueError, "dev has"):
            data.select_cohorts(users_for("train", 10), caps={"train": 1, "dev": 1, "test": 0},
                                selector_count=1)
        with self.assertRaisesRegex(ValueError, "selector"):
            data.select_cohorts([], caps={"train": 0, "dev": 0, "test": 0}, selector_count=1)


class ChronologyTests(unittest.TestCase):
    def test_existing_metadata_membership_index_is_reused(self):
        class TrackingSet(set):
            def __init__(self, values):
                super().__init__(values)
                self.membership_calls = 0

            def __contains__(self, value):
                self.membership_calls += 1
                return super().__contains__(value)

        known = TrackingSet(event.item_id for event in events())
        result, _ = contexts(known=known)
        self.assertEqual(len(result), 14)
        self.assertGreater(known.membership_calls, 0)

    def test_recent_history_strict_and_all_past_seen(self):
        result, audit = contexts()
        self.assertEqual(len(result), 14)
        last = result[-1]
        self.assertEqual([entry.item_id for entry in last.history], [f"book-{i}" for i in range(4, 14)])
        self.assertEqual(last.seen_item_ids, tuple(sorted(f"book-{i}" for i in range(14))))
        self.assertTrue(all(entry.timestamp < last.target_timestamp for entry in last.history))
        self.assertEqual(audit["legal_queries"], 14)

    def test_low_and_unknown_target_rating_retained(self):
        source = (data.TimelineEvent("first", 4, 1), data.TimelineEvent("second", 1, 2),
                  data.TimelineEvent("third", None, 3))
        result, _ = contexts(source=source)
        self.assertEqual([context.target_item_id for context in result], ["second", "third"])
        self.assertEqual([entry.rating for entry in result[-1].history], [4, 1])

    def test_missing_target_does_not_skip_to_next_book(self):
        source = events(4)
        result, audit = contexts(source=source, known={"book-0", "book-2", "book-3"})
        self.assertFalse(result)
        self.assertEqual(audit["missing_target_metadata"], 1)
        self.assertEqual(audit["missing_history_metadata"], 2)

    def test_missing_history_does_not_backfill_or_delete(self):
        result, audit = contexts(source=events(15), known={f"book-{i}" for i in range(15)} - {"book-12"})
        self.assertEqual(result[-1].target_item_id, "book-11")
        self.assertEqual(audit["missing_target_metadata"], 1)
        self.assertEqual(audit["missing_history_metadata"], 2)

    def test_dev_last_only_no_fallback_for_missing_target(self):
        result, audit = contexts("dev", source=events(6), known={f"book-{i}" for i in range(5)})
        self.assertFalse(result)
        self.assertEqual(audit["positions_considered"], 1)
        self.assertEqual(audit["missing_target_metadata"], 1)

    def test_unique_time_group_required_no_arbitrary_target_order(self):
        source = (data.TimelineEvent("a", 2, 1), data.TimelineEvent("b", 5, 2),
                  data.TimelineEvent("c", 4, 2), data.TimelineEvent("d", 3, 3))
        result, audit = contexts(source=source)
        self.assertFalse(result)
        self.assertEqual(audit["multiple_immediate_next_items"], 1)
        self.assertEqual(audit["tied_history_timestamps"], 1)

    def test_history_limit_cannot_cut_time_group(self):
        source = (data.TimelineEvent("a", 2, 1), data.TimelineEvent("b", 5, 1),
                  data.TimelineEvent("c", 4, 2))
        result, audit = contexts(source=source, history_limit=1)
        self.assertFalse(result)
        self.assertEqual(audit["history_cut_through_time_group"], 1)

    def test_exact_duplicates_collapse_conflicts_and_unknowns_do_not(self):
        source = (data.TimelineEvent("a", 2, 1), data.TimelineEvent("a", 2, 1),
                  data.TimelineEvent("a", 5, 1), data.TimelineEvent("b", 4, 2))
        result, audit = contexts(source=source)
        self.assertFalse(result)
        self.assertEqual(audit["exact_duplicate_rows_collapsed"], 1)
        self.assertEqual(audit["unknown_or_conflicting_history_rating"], 1)
        source = (data.TimelineEvent("a", None, 1), data.TimelineEvent("b", 4, 2))
        self.assertFalse(contexts(source=source)[0])

    def test_revisit_not_new_target_and_not_deleted_from_timeline(self):
        source = (data.TimelineEvent("a", 2, 1), data.TimelineEvent("b", 5, 2),
                  data.TimelineEvent("a", 4, 3), data.TimelineEvent("c", 3, 4))
        result, audit = contexts(source=source)
        self.assertEqual([context.target_item_id for context in result], ["b", "c"])
        self.assertEqual([entry.item_id for entry in result[-1].history], ["a", "b", "a"])
        self.assertEqual(audit["revisited_target"], 1)

    def test_quantile_query_selection_distinct_order_invariant(self):
        full = contexts(source=events(22))[0]
        chosen = data.select_temporal_queries(full, limit=5)
        self.assertEqual(chosen, data.select_temporal_queries(tuple(reversed(full)), limit=5))
        self.assertEqual(len({context.query_id for context in chosen}), 5)
        self.assertEqual([context.target_timestamp for context in chosen], [103, 107, 111, 115, 119])
        self.assertEqual(data.select_temporal_queries(full[:3]), full[:3])
        with self.assertRaisesRegex(ValueError, "unique"):
            data.select_temporal_queries((full[0], full[0]))


class CandidateTests(unittest.TestCase):
    def test_exact_positive_random_and_hard_deterministic_group(self):
        context = contexts()[0][-1]
        catalog = [*context.seen_item_ids, context.target_item_id, *[f"other-{i}" for i in range(12)]]
        retrieval = ["book-0", "other-0", "other-0", context.target_item_id, "other-1", "other-2"]
        first = data.training_candidates(context, catalog_ids=catalog, retrieved_ids=retrieval)
        second = data.training_candidates(context, catalog_ids=reversed(catalog), retrieved_ids=retrieval)
        self.assertEqual(first, second)
        self.assertEqual(len(first.candidate_ids), 8)
        self.assertTrue({"other-0", "other-1", "other-2"} <= set(first.candidate_ids))
        self.assertEqual(first.candidate_ids[first.positive_index], context.target_item_id)
        self.assertFalse(set(context.seen_item_ids) & set(first.candidate_ids))

    def test_future_items_are_not_filtered_by_sampler(self):
        full = contexts(source=events(15))[0]
        context = full[0]
        catalog = [f"book-{i}" for i in range(15)]
        query = data.training_candidates(context, catalog_ids=catalog,
                                         retrieved_ids=["book-2", "book-3", "book-4"])
        self.assertTrue({"book-2", "book-3", "book-4"} <= set(query.candidate_ids))

    def test_no_random_fallback_for_missing_actual_hard_input(self):
        context = contexts()[0][-1]
        catalog = [context.target_item_id, *[f"other-{i}" for i in range(20)]]
        with self.assertRaisesRegex(ValueError, "three"):
            data.training_candidates(context, catalog_ids=catalog, retrieved_ids=["other-0"])
        with self.assertRaisesRegex(ValueError, "four"):
            data.training_candidates(context, catalog_ids=catalog[:7], retrieved_ids=catalog[1:4])

    def test_reusable_sampler_matches_top200_sampling_reference(self):
        context = contexts()[0][-1]
        catalog = {*context.seen_item_ids, context.target_item_id, *[f"other-{i}" for i in range(40)]}
        retrieved = ["book-0", context.target_item_id, "other-0", "other-0", *[f"other-{i}" for i in range(1, 20)]]
        excluded = set(context.seen_item_ids) | {context.target_item_id}
        hard_pool = list(dict.fromkeys(item for item in retrieved[:200] if item in catalog and item not in excluded))
        samples = []
        for seed in (0, 17, 42, 73):
            rng_seed = int(hashlib.sha256(
                f"{data.SCHEMA_VERSION}|candidates|{seed}|{context.query_id}".encode()).hexdigest(), 16)
            rng = random.Random(rng_seed)
            hard = rng.sample(hard_pool, 3)
            self.assertEqual(len(set(hard)), 3)
            samples.append(tuple(hard))
            pool = sorted(catalog - excluded - set(hard))
            expected = [context.target_item_id, *rng.sample(pool, 4), *hard]
            rng.shuffle(expected)
            sampler = data.TrainingCandidateSampler(catalog, seed=seed)
            self.assertEqual(sampler.sample(context, retrieved_ids=retrieved).candidate_ids, tuple(expected))
        self.assertTrue(any(set(sample) != set(hard_pool[:3]) for sample in samples))
        self.assertGreater(len(set(samples)), 1)

    def test_hard_pool_uses_first200_positions_before_deduplication(self):
        context = contexts()[0][-1]
        catalog = [context.target_item_id, *[f"other-{i}" for i in range(20)]]
        sampler = data.TrainingCandidateSampler(catalog)
        top200 = [context.seen_item_ids[0]] * 197 + ["other-0", "other-1", "other-2"]
        first = sampler.sample(context, retrieved_ids=top200 + ["other-3"])
        changed_tail = sampler.sample(context, retrieved_ids=top200 + ["other-4", "other-5"])
        self.assertEqual(first, changed_tail)
        # The third legal hard item at position 201 cannot rescue Top-200.
        with self.assertRaisesRegex(ValueError, "Top-200"):
            sampler.sample(context, retrieved_ids=[context.seen_item_ids[0]] * 198
                           + ["other-0", "other-1", "other-2"])

    def test_reusable_sampler_does_not_reiterate_large_catalog(self):
        context = contexts()[0][-1]

        class OnceCatalog:
            def __init__(self):
                self.calls = 0

            def __iter__(self):
                self.calls += 1
                if self.calls != 1:
                    raise AssertionError("catalog must be indexed once")
                return iter([context.target_item_id, *[f"other-{i:06d}" for i in range(10_000)]])

        class NoIterationTuple(tuple):
            def __iter__(self):
                raise AssertionError("query must not iterate indexed catalog")

        catalog = OnceCatalog()
        sampler = data.TrainingCandidateSampler(catalog)
        sampler._catalog_ids = NoIterationTuple(sampler._catalog_ids)
        expected = sampler.sample(context, retrieved_ids=["other-000001", "other-000002", "other-000003"])
        for _ in range(5):
            self.assertEqual(sampler.sample(context, retrieved_ids=["other-000001", "other-000002", "other-000003"]),
                             expected)
        self.assertEqual(catalog.calls, 1)

    def test_index_mapping_handles_dense_exclusions_without_rejection(self):
        context = contexts()[0][-1]
        # Exactly four random items remain; all must be selected.
        hard = ["other-01", "other-03", "other-05"]
        random_items = ["other-02", "other-04", "other-06", "other-08"]
        extra_seen = tuple(f"other-{i:02d}" for i in range(10)
                           if f"other-{i:02d}" not in hard + random_items)
        context = replace(context, seen_item_ids=tuple(sorted((*context.seen_item_ids, *extra_seen))))
        catalog = [context.target_item_id, *context.seen_item_ids, *hard, *random_items]
        sampler = data.TrainingCandidateSampler(catalog)
        query = sampler.sample(context, retrieved_ids=hard)
        self.assertEqual(set(query.candidate_ids), {context.target_item_id, *hard, *random_items})
        self.assertFalse(set(query.candidate_ids) & set(context.seen_item_ids))

    def test_retrieval_miss_kept_and_target_never_injected(self):
        context = contexts("dev")[0][0]
        query = data.evaluation_candidates(context, ["book-0", "x", "x", "y"], count=2)
        self.assertEqual(query.candidate_ids, ("x", "y"))
        self.assertIsNone(query.positive_index)
        self.assertFalse(query.retrieval_hit)
        hit = data.evaluation_candidates(context, ["x", context.target_item_id], count=2)
        self.assertEqual(hit.positive_index, 1)
        with self.assertRaisesRegex(ValueError, "requested"):
            data.evaluation_candidates(context, ["x", "x"], count=2)


class SchemaAndSealingTests(unittest.TestCase):
    def test_group_round_trip_loader_and_candidate_identity(self):
        query = group()
        self.assertEqual(query, data.RecommendationQuery.from_dict(query.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            data.write_queries(path, (query,))
            self.assertEqual(data.read_queries(path, split="train"), (query,))

    def test_invalid_labels_and_history_fail(self):
        query = group()
        with self.assertRaisesRegex(ValueError, "positive_index"):
            replace(query, positive_index=None)
        with self.assertRaisesRegex(ValueError, "target item"):
            replace(query, positive_index=(query.positive_index + 1) % 8)
        with self.assertRaisesRegex(ValueError, "unique"):
            replace(query, candidate_ids=("x", "x"))
        with self.assertRaisesRegex(ValueError, "all past"):
            replace(query, candidate_ids=(query.target_item_id, query.seen_item_ids[0]), positive_index=0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            replace(query, history=(replace(query.history[0], timestamp=query.target_timestamp),))
        for rating in (0, 6, True, 1.5):
            with self.assertRaises(ValueError):
                data.HistoryEntry("book", rating, 1)
        with self.assertRaisesRegex(ValueError, "fields"):
            data.RecommendationQuery.from_dict({**query.to_dict(), "label": 4})

    def test_sealed_test_rejected_before_file_open(self):
        with patch.object(Path, "open", side_effect=AssertionError("must not open")):
            with self.assertRaisesRegex(ValueError, "sealed"):
                data.read_queries("sealed-test-labels.jsonl", split="test")
            with self.assertRaisesRegex(ValueError, "sealed"):
                data.load_timeline_csv("sealed.csv", users_for("test", 1), split="test")
        with self.assertRaisesRegex(ValueError, "split boundary"):
            data.load_timeline_csv("not-opened.csv", users_for("test", 1), split="train")
        row = group().to_dict()
        row["user_id"] = users_for("test", 1)[0]
        with self.assertRaisesRegex(ValueError, "hash split"):
            data.RecommendationQuery.from_dict(row)

    def test_duplicate_queries_and_mixed_split_fail(self):
        query = group()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            with self.assertRaisesRegex(ValueError, "duplicate"):
                data.write_queries(path, (query, query))
            self.assertFalse(path.exists())
            path.write_text(json.dumps(query.to_dict()) + "\n" + json.dumps(query.to_dict()) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                data.read_queries(path, split="train")
            data.write_queries(path, (group("dev"),))
            with self.assertRaisesRegex(ValueError, "split"):
                data.read_queries(path, split="train")

    def test_failed_write_preserves_existing_dataset(self):
        query = group()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            data.write_queries(path, (query,))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "mix"):
                data.write_queries(path, (query, group("dev")))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "empty"):
                data.write_queries(path, ())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_csv_ignores_unselected_test_values_and_keeps_unknown_rating(self):
        train = users_for("train", 1)[0]
        sealed = users_for("test", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("user_id", "parent_asin", "rating", "timestamp"))
                writer.writerow((sealed, "", "FORBIDDEN", "FORBIDDEN"))
                writer.writerow((train, "a", "3.000", "1"))
                writer.writerow((train, "b", "1.000000000000000001", "2"))
            result = data.load_timeline_csv(path, [train], split="train")
            self.assertEqual(result[train], (data.TimelineEvent("a", 3, 1), data.TimelineEvent("b", None, 2)))
            with path.open("a") as stream:
                stream.write(f"{train},c,4,UNKNOWN\n")
            with self.assertRaisesRegex(ValueError, "timestamp"):
                data.load_timeline_csv(path, [train], split="train")


if __name__ == "__main__":
    unittest.main()
